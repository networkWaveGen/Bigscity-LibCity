"""LibCity's two-partition GeomGCN autoencoder, without mutable DGL graphs.

The LibCity variant partitions observed road edges and self edges; it does not
construct the paper's embedding-space neighborhoods. Each partition has its own
projection, and source/destination normalization uses the complete graph degree.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.road_representation.runtime_reconstruction import graph_matrix


class GeometricHead(nn.Module):
    def __init__(self, inputs, outputs, dropout, merge, activation):
        super().__init__()
        self.projections = nn.ModuleList(nn.Linear(inputs, outputs, bias=False) for _ in range(2))
        for projection in self.projections:
            nn.init.xavier_uniform_(projection.weight)
        self.dropout = nn.Dropout(dropout)
        self.merge, self.activation = merge, activation
        self.output_dim = outputs * (2 if merge == 'cat' else 1)

    def forward(self, features, edges, norm):
        features = self.dropout(features)
        source, destination = edges
        projected = self.projections[0](features) * norm
        neighbors = projected.new_zeros(projected.shape).index_add(0, destination, projected[source])
        own = self.projections[1](features) * norm
        merged = torch.cat((neighbors, own), -1) if self.merge == 'cat' else (neighbors + own) / 2
        merged = merged * norm
        return F.relu(merged) if self.activation else merged


class GeometricLayer(nn.Module):
    def __init__(self, inputs, outputs, heads, dropout, division_merge, channel_merge, activation=False):
        super().__init__()
        self.heads = nn.ModuleList(GeometricHead(inputs, outputs, dropout, division_merge, activation)
                                  for _ in range(heads))
        self.channel_merge = channel_merge
        self.output_dim = self.heads[0].output_dim * (heads if channel_merge == 'cat' else 1)

    def forward(self, features, edges, norm):
        channels = [head(features, edges, norm) for head in self.heads]
        return torch.cat(channels, -1) if self.channel_merge == 'cat' else torch.stack(channels).mean(0)


class GeomGCN(nn.Module):
    def __init__(self, *, metadata, hidden_dim=144, embedding_dim=32, dropout=.5,
                 num_heads_layer_one=1, num_heads_layer_two=1,
                 layer_one_ggcn_merge='cat', layer_one_channel_merge='cat',
                 layer_two_ggcn_merge='mean', layer_two_channel_merge='mean'):
        super().__init__()
        dimensions = (hidden_dim, embedding_dim, num_heads_layer_one, num_heads_layer_two)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError('GeomGCN dimensions and head counts must be positive integers')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('GeomGCN dropout must be finite and in [0, 1)')
        if any(value not in ('cat', 'mean') for value in (layer_one_ggcn_merge, layer_one_channel_merge,
                                                       layer_two_ggcn_merge, layer_two_channel_merge)):
            raise ValueError('GeomGCN partition and channel merges must be cat or mean')
        if metadata.num_nodes < 1 or metadata.feature_dim < 1:
            raise ValueError('GeomGCN needs nodes and observed node attributes')
        graph = graph_matrix(metadata).tocoo()
        selected = (graph.row != graph.col) & (graph.data != 0)
        edges = np.unique(np.stack((graph.row[selected], graph.col[selected])), axis=1)
        edge_index = torch.tensor(edges, dtype=torch.long)
        # Every node receives exactly one self edge, regardless of input loops.
        degree = torch.bincount(edge_index[1], minlength=metadata.num_nodes).float() + 1
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('norm', degree.rsqrt()[:, None])
        self.geomgcn1 = GeometricLayer(metadata.feature_dim, hidden_dim, num_heads_layer_one, dropout,
                                      layer_one_ggcn_merge, layer_one_channel_merge, activation=True)
        self.geomgcn2 = GeometricLayer(self.geomgcn1.output_dim, embedding_dim, num_heads_layer_two, dropout,
                                      layer_two_ggcn_merge, layer_two_channel_merge)
        self.embedding_width = self.geomgcn2.output_dim
        self.geomgcn3 = GeometricLayer(self.embedding_width, self.geomgcn1.output_dim, num_heads_layer_two,
                                      dropout, layer_two_ggcn_merge, layer_two_channel_merge)
        # The reconstruction must have one value per input attribute. The old
        # concatenating second-layer settings broke decoder dimensions; retain
        # their actual encoder widths but average the final readout partitions.
        self.geomgcn4 = GeometricLayer(self.geomgcn3.output_dim, metadata.feature_dim, num_heads_layer_one,
                                      dropout, 'mean', 'mean', activation=True)

    def encode(self, features):
        return self.geomgcn2(self.geomgcn1(features, self.edge_index, self.norm), self.edge_index, self.norm)

    def forward(self, batch):
        hidden = self.geomgcn3(self.encode(batch['node_features']), self.edge_index, self.norm)
        prediction = self.geomgcn4(hidden, self.edge_index, self.norm)
        return prediction[batch['node_indices']] if 'node_indices' in batch else prediction
