"""Road graph autoencoders: modules own tensors, never scalers or output paths."""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.utils import get_supports_matrix


def graph_matrix(metadata):
    if metadata.adjacency is not None:
        return sp.coo_matrix(metadata.adjacency)
    edges = np.asarray(metadata.info['edge_index'])
    return sp.coo_matrix((np.ones(edges.shape[1],dtype=np.float32),(edges[0],edges[1])),
                         shape=(metadata.num_nodes,metadata.num_nodes))


class ChebyshevLayer(nn.Module):
    def __init__(self, adjacency, inputs, outputs, order, filter_type):
        super().__init__()
        with np.errstate(divide='ignore',invalid='ignore'):
            graph = adjacency.copy() if sp.issparse(adjacency) else np.array(adjacency,copy=True)
            supports = get_supports_matrix(graph,filter_type)
        self.order,self.support_count = order,len(supports)
        for index,support in enumerate(supports):
            support = support.tocoo()
            graph = torch.sparse_coo_tensor(np.stack((support.row,support.col)),support.data,
                support.shape,dtype=torch.float32,check_invariants=True).coalesce()
            self.register_buffer(f'support_{index}',graph)
        self.weight = nn.Parameter(torch.empty(inputs*(1+order*len(supports)),outputs))
        self.biases = nn.Parameter(torch.zeros(outputs))
        nn.init.xavier_normal_(self.weight)

    def forward(self, source):
        # Sparse mm's CPU backward requires matching FP32/FP64 operands; the
        # dense learned projection below still follows the caller's autocast.
        with torch.autocast(source.device.type,enabled=False):
            source = source.to(self.weight.dtype)
            terms = [source]
            if self.order:
                for index in range(self.support_count):
                    support = getattr(self,f'support_{index}')
                    previous = source
                    following = torch.sparse.mm(support,source)
                    terms.append(following)
                    for _ in range(2,self.order+1):
                        current = 2*torch.sparse.mm(support,following)-previous
                        terms.append(current)
                        previous,following = following,current
            expanded = torch.stack(terms,dim=-1).flatten(1)
        return expanded@self.weight+self.biases


class ChebConv(nn.Module):
    def __init__(self,*,metadata,embedding_dim=32,max_diffusion_step=2,filter_type='dual_random_walk'):
        super().__init__()
        if embedding_dim<1 or max_diffusion_step<0 or filter_type not in ('laplacian','random_walk','dual_random_walk'):
            raise ValueError('ChebConv requires positive embedding_dim, nonnegative order and a supported filter_type')
        graph = graph_matrix(metadata)
        self.encoder = ChebyshevLayer(graph,metadata.feature_dim,embedding_dim,max_diffusion_step,filter_type)
        self.decoder = ChebyshevLayer(graph,embedding_dim,metadata.feature_dim,max_diffusion_step,filter_type)

    def encode(self,features):
        return self.encoder(features)

    def forward(self,batch):
        prediction = self.decoder(self.encode(batch['node_features']))
        return prediction[batch['node_indices']] if 'node_indices' in batch else prediction


class AttentionLayer(nn.Module):
    def __init__(self,inputs,outputs,heads,dropout):
        super().__init__()
        self.heads,self.outputs = heads,outputs
        self.linear_proj = nn.Linear(inputs,heads*outputs,bias=False)
        self.scoring_fn_target = nn.Parameter(torch.empty(1,heads,outputs))
        self.scoring_fn_source = nn.Parameter(torch.empty(1,heads,outputs))
        self.bias = nn.Parameter(torch.zeros(outputs))
        self.skip_proj = nn.Linear(inputs,heads*outputs,bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

    def forward(self,source,edge_index):
        source = self.dropout(source)
        projected = self.dropout(self.linear_proj(source).reshape(-1,self.heads,self.outputs))
        origin,destination = edge_index
        aggregate = projected.new_zeros(projected.shape)
        if origin.numel():
            left = (projected*self.scoring_fn_source).sum(-1)[origin]
            right = (projected*self.scoring_fn_target).sum(-1)[destination]
            scores = F.leaky_relu(left+right,.2)
            # Normalize within each target neighborhood/head, preventing remote
            # high-score edges from underflowing an unrelated neighborhood.
            maxima = scores.new_full((len(source),self.heads),-torch.inf)
            maxima.scatter_reduce_(0,destination[:,None].expand_as(scores),scores.detach(),reduce='amax')
            exponential = torch.exp(scores-maxima[destination])
            denominator = exponential.new_zeros((len(source),self.heads)).index_add(0,destination,exponential)
            attention = self.dropout((exponential/denominator[destination]).unsqueeze(-1))
            weighted = projected[origin]*attention
            aggregate = weighted.new_zeros(projected.shape)
            aggregate.index_add_(0,destination,weighted)
        if self.outputs==source.shape[-1]:
            aggregate = aggregate+source[:,None]
        else:
            aggregate = aggregate+self.skip_proj(source).reshape(-1,self.heads,self.outputs)
        return F.elu(aggregate.mean(1)+self.bias)


class GAT(nn.Module):
    def __init__(self,*,metadata,embedding_dim=32,num_heads=5,dropout=.6):
        super().__init__()
        if min(embedding_dim,num_heads)<1 or not 0<=dropout<1:
            raise ValueError('GAT requires positive embedding/head dimensions and dropout in [0,1)')
        graph = graph_matrix(metadata)
        self.register_buffer('edge_index',torch.tensor(np.stack((graph.row,graph.col)),dtype=torch.long))
        self.encoder = AttentionLayer(metadata.feature_dim,embedding_dim,num_heads,dropout)
        self.decoder = AttentionLayer(embedding_dim,metadata.feature_dim,num_heads,dropout)

    def encode(self,features):
        return self.encoder(features,self.edge_index)

    def forward(self,batch):
        prediction = self.decoder(self.encode(batch['node_features']),self.edge_index)
        return prediction[batch['node_indices']] if 'node_indices' in batch else prediction
