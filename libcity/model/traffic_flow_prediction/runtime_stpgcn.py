"""Registered position-aware graph layers with factored Gaussian relations."""
import numpy as np
import torch
from torch import nn


def bounded_hops(graph, maximum):
    """Shortest outgoing hop counts; -1 denotes unreachable or beyond the bound."""
    nodes = len(graph)
    neighbors = [np.flatnonzero(row>0).tolist() for row in graph]
    distance = np.full((nodes,nodes),-1,dtype=np.int64)
    for origin in range(nodes):
        distance[origin,origin] = 0
        frontier = [origin]
        for step in range(1,maximum+1):
            following = []
            for node in frontier:
                for neighbor in neighbors[node]:
                    if distance[origin,neighbor]<0:
                        distance[origin,neighbor] = step
                        following.append(neighbor)
            frontier = following
            if not frontier:
                break
    return distance


class GaussianComponent(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(1,width))
        self.inv_sigma = nn.Parameter(torch.randn(1,width))

    def forward(self, embedding):
        embedding = embedding.float() if embedding.dtype in (torch.float16,torch.bfloat16) else embedding
        return -.5*((embedding-self.mu).square()*self.inv_sigma.square()).sum(-1)


class RelationInference(nn.Module):
    def __init__(self, width, beta):
        super().__init__()
        self.beta = beta
        self.components = nn.ModuleList(GaussianComponent(width) for _ in range(6))

    def forward(self, spatial, temporal, relative_space, relative_time, mask):
        g = self.components
        # Retain LibCity's source/destination convention before its final transpose.
        spatial_score = g[0](spatial)[:,None]+g[1](spatial)[None,:]+g[4](relative_space)
        spatial_weight = spatial_score.T.exp()*mask
        padded = torch.cat((temporal.new_zeros(temporal.shape[0],self.beta,1,temporal.shape[-1]),temporal),1)
        past = padded.unfold(1,self.beta+1,1).permute(0,1,4,2,3)
        temporal_score = g[2](temporal).squeeze(-1)[...,None]+g[3](past).squeeze(-1)+g[5](relative_time).squeeze(-1)
        return spatial_weight,temporal_score.exp()


class PositionGraphConv(nn.Module):
    def __init__(self, channels, embedding):
        super().__init__()
        self.x_proj = nn.Linear(channels,2*channels)
        self.se_proj = nn.Linear(embedding,2*channels,bias=False)
        self.te_proj = nn.Linear(embedding,2*channels,bias=False)
        self.ln = nn.LayerNorm(2*channels)

    def forward(self, aggregated, spatial, temporal):
        value = self.x_proj(aggregated)+self.se_proj(spatial)+self.te_proj(temporal)
        left,right = self.ln(value).chunk(2,-1)
        return left*right.sigmoid()


class GLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Conv2d(channels,2*channels,1)

    def forward(self, value):
        left,right = self.linear(value).chunk(2,1)
        return left*right.sigmoid()


class FeatureSelection(nn.Module):
    def __init__(self, history, channels):
        super().__init__()
        self.fc = nn.Conv2d(history,channels,(1,channels))
        self.glu = GLU(channels)

    def forward(self, value):
        return self.glu(self.fc(value))


class PositionGraphLayer(nn.Module):
    def __init__(self, history, channels, embedding, beta):
        super().__init__()
        self.beta = beta
        self.relation = RelationInference(embedding,beta)
        self.convolution = PositionGraphConv(channels,embedding)
        self.selection = FeatureSelection(history,channels)

    def forward(self, value, spatial, temporal, relative_space, relative_time, mask):
        spatial_weight,temporal_weight = self.relation(spatial,temporal,relative_space,relative_time,mask)
        # exp(spatial_log + temporal_log) factors exactly. Aggregate space once
        # for the full history, then apply the causal per-query temporal weights.
        filtered = spatial_weight@value
        padded = torch.cat((filtered.new_zeros(filtered.shape[0],self.beta,*filtered.shape[2:]),filtered),1)
        past = padded.unfold(1,self.beta+1,1).permute(0,1,4,2,3)
        aggregated = (past*temporal_weight[...,None,None]).sum(2)
        return self.convolution(aggregated,spatial,temporal)


class STPGCN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, C=64, L=3, d=8, alpha=4, beta=2):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,C=C,L=L,d=d,
                num_nodes=metadata.num_nodes,feature_dim=metadata.feature_dim,output_dim=metadata.output_dim).items():
            if type(value) is not int or value<1:
                raise ValueError(f'STPGCN {name} must be a positive integer')
        if any(type(value) is not int or value<0 for value in (alpha,beta)):
            raise ValueError('STPGCN alpha and beta must be nonnegative integers')
        layout = metadata.input_layout.get('history_features',{})
        if (metadata.input_layout.get('kind')!='window' or layout.get('kind')!='calendar_positions'
                or tuple(layout.get('columns',()))!=('weekday','time_slot')):
            raise ValueError('STPGCN requires window data with integer weekday/time_slot history_features')
        day_length = layout.get('steps_per_day')
        if type(day_length) is not int or day_length<1:
            raise ValueError('STPGCN steps_per_day must be a positive integer')
        self.input_window,self.output_window = input_window,output_window
        self.num_nodes,self.feature_dim,self.output_dim = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.day_length = day_length
        graph = np.array(metadata.adjacency,dtype=np.float32,copy=True)
        if graph.shape!=(self.num_nodes,self.num_nodes) or not np.isfinite(graph).all() or np.any(graph<0):
            raise ValueError('STPGCN requires a finite nonnegative square adjacency matching num_nodes')
        hops = bounded_hops(graph,alpha)
        self.register_buffer('hop_distance',torch.from_numpy(np.where(hops<0,alpha+1,hops)))
        self.register_buffer('range_mask',torch.from_numpy(hops>=0))
        self.spatial_embedding = nn.Embedding(self.num_nodes,d)
        self.week_embedding = nn.Embedding(7,d)
        self.day_embedding = nn.Embedding(day_length,d)
        self.spatial_relative_embedding = nn.Embedding(alpha+2,d)
        self.temporal_relative_embedding = nn.Embedding(beta+1,d)
        self.input_projection = nn.Linear(self.feature_dim,C)
        self.initial_selection = FeatureSelection(input_window,C)
        self.layers = nn.ModuleList(PositionGraphLayer(input_window,C,d,beta) for _ in range(L))
        self.fusion = GLU(C*(L+1))
        self.output_projection = nn.Conv2d(C*(L+1),output_window*self.output_dim,1)

    def forward(self, inputs):
        source = inputs['X']
        if source.ndim!=4 or source.shape[1:]!=(self.input_window,self.num_nodes,self.feature_dim):
            raise ValueError('STPGCN X must have configured [batch,history,node,feature] dimensions')
        positions = inputs.get('history_features')
        if (positions is None or positions.dtype!=torch.long or positions.shape!=(len(source),self.input_window,2)):
            raise ValueError('STPGCN history_features must be int64 [batch,history,2] weekday/time_slot positions')
        weekday,slot = positions.unbind(-1)
        if torch.any((weekday<0)|(weekday>=7)|(slot<0)|(slot>=self.day_length)):
            raise ValueError('STPGCN calendar positions are outside weekday/time-slot bounds')
        spatial = self.spatial_embedding.weight
        temporal = (self.week_embedding(weekday)+self.day_embedding(slot)).unsqueeze(2)
        relative_space = self.spatial_relative_embedding(self.hop_distance)
        relative_time = self.temporal_relative_embedding.weight[:,None]
        value = self.input_projection(source)
        skips = [self.initial_selection(value)]
        for layer in self.layers:
            value = layer(value,spatial,temporal,relative_space,relative_time,self.range_mask)
            skips.append(layer.selection(value))
        prediction = self.output_projection(self.fusion(torch.cat(skips,1)))
        return prediction.reshape(len(source),self.output_window,self.output_dim,self.num_nodes).transpose(2,3)
