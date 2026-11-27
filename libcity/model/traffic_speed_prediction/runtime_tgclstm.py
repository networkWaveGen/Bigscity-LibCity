"""Graph-filtered LSTM with immutable supports and label-free rolling inputs."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class FilterLinear(nn.Module):
    def __init__(self,graph,features,outputs):
        super().__init__()
        self.register_buffer('filter_square_matrix',graph.repeat(outputs,features))
        self.weight = nn.Parameter(torch.empty(outputs*len(graph),features*len(graph)))
        nn.init.uniform_(self.weight,-1/math.sqrt(self.weight.shape[0]),1/math.sqrt(self.weight.shape[0]))

    def forward(self,source):
        return F.linear(source,self.weight*self.filter_square_matrix)


class TGCLSTM(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,hops=3,clamp_adjacency=True):
        super().__init__()
        if hops<1 or metadata.input_layout.get('kind')!='window':
            raise ValueError('TGCLSTM requires positive hops and contiguous input windows')
        self.nodes,self.features,self.outputs = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.horizon = output_window
        self.future_indices = tuple(metadata.input_layout.get('future_feature_indices',()))
        graph = torch.tensor(metadata.adjacency.copy(),dtype=torch.float32)
        reachability = metadata.input_layout.get('reachability')
        if reachability is not None:
            graph = (graph>1e-4).to(graph.dtype)
            reachability = torch.tensor(reachability.copy(),dtype=graph.dtype)
        support = torch.eye(self.nodes)
        supports = []
        for _ in range(hops):
            support = support@graph
            if clamp_adjacency:
                support = support.clamp_max(1.)
            supports.append(support if reachability is None else support*reachability)
        self.register_buffer('supports',torch.stack(supports))
        self.register_buffer('cell_filter',supports[-1].repeat(self.outputs,self.outputs))
        self.gc_list = nn.ModuleList(FilterLinear(matrix,self.features,self.outputs) for matrix in supports)
        width = self.nodes*self.outputs
        self.fl,self.il,self.ol,self.Cl = (nn.Linear(width*(hops+1),width) for _ in range(4))
        self.Neighbor_weight = nn.Parameter(torch.empty(width,width))
        nn.init.uniform_(self.Neighbor_weight,-1/math.sqrt(width),1/math.sqrt(width))

    def one_step(self,source):
        hidden = source.new_zeros(source.shape[0],self.nodes*self.outputs)
        cell = torch.zeros_like(hidden)
        for position in range(source.shape[1]):
            value = source[:,position].transpose(1,2).reshape(source.shape[0],-1)
            filtered = torch.cat([layer(value) for layer in self.gc_list],dim=-1)
            combined = torch.cat((filtered,hidden),dim=-1)
            cell = torch.sigmoid(self.fl(combined))*(cell@(self.cell_filter*self.Neighbor_weight)) \
                +torch.sigmoid(self.il(combined))*torch.tanh(self.Cl(combined))
            hidden = torch.sigmoid(self.ol(combined))*torch.tanh(cell)
        return hidden.reshape(source.shape[0],self.outputs,self.nodes).transpose(1,2)

    def forward(self,batch):
        source = batch['X']
        predictions = []
        for position in range(self.horizon):
            prediction = self.one_step(source)
            predictions.append(prediction)
            following = source[:,-1].clone()
            following[...,:self.outputs] = prediction
            if self.future_indices:
                if 'future_features' not in batch:
                    raise ValueError('TGCLSTM requires explicit known future_features for declared future channels')
                following[...,list(self.future_indices)] = batch['future_features'][:,position]
            source = torch.cat((source[:,1:],following[:,None]),dim=1)
        return torch.stack(predictions,dim=1)
