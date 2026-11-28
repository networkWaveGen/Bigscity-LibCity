"""Dual graph/self recurrent streams with sample-local variance fusion."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FilterLinear(nn.Module):
    def __init__(self,graph,features,outputs):
        super().__init__()
        self.register_buffer('filter_square_matrix',graph.repeat(outputs,features))
        self.weight = nn.Parameter(torch.empty(len(graph)*outputs,len(graph)*features))
        bound = 1/math.sqrt(self.weight.shape[1])
        nn.init.uniform_(self.weight,-bound,bound)

    def forward(self,source):
        return F.linear(source,self.weight*self.filter_square_matrix)


def sample_variance(values):
    # Retain sample variance for nondegenerate streams. A scalar stream has no
    # empirical dispersion; using correction=1 there would return NaN.
    if values.dtype in (torch.float16,torch.bfloat16):
        values = values.float()
    return values.var(dim=-1,keepdim=True,correction=int(values.shape[-1]>1))


def fuse_streams(graph_hidden,self_hidden,observations,filtered,coefficient):
    direct,neighbors = sample_variance(observations),sample_variance(filtered)
    tiny = torch.finfo(direct.dtype).tiny
    scale = torch.maximum(direct,neighbors).clamp_min(tiny)
    a,b = direct/scale,neighbors/scale
    # Preserve the authors' asymmetric coefficient formula. Scaling both
    # variances by the same value cancels algebraically, avoiding tiny divisors.
    result = (graph_hidden*a*coefficient+self_hidden*b)/(a+b*coefficient).clamp_min(tiny)
    return torch.where((direct+neighbors)>0,result,(graph_hidden+self_hidden)/2)


class DKFN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,hops=3,clamp_adjacency=False,threshold=1e-4):
        super().__init__()
        if metadata.input_layout.get('kind')!='window':
            raise ValueError('DKFN requires contiguous window sampling')
        if type(hops) is not int or hops<1 or type(clamp_adjacency) is not bool:
            raise ValueError('DKFN requires positive hops and boolean clamp_adjacency')
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or threshold<0:
            raise ValueError('DKFN threshold must be finite and nonnegative')
        self.nodes,self.features,self.outputs = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.horizon,self.length = output_window,input_window
        self.future_indices = tuple(metadata.input_layout.get('future_feature_indices',()))
        adjacency = np.array(metadata.adjacency,dtype=np.float32,copy=True)
        if adjacency.shape!=(self.nodes,self.nodes) or not np.isfinite(adjacency).all() or np.any(adjacency<0):
            raise ValueError('DKFN adjacency must be finite, nonnegative and match nodes')
        graph = torch.tensor(adjacency>threshold,dtype=torch.float32)
        degree = graph.sum(0)
        inverse = torch.where(degree>0,degree.clamp_min(1).reciprocal(),torch.zeros_like(degree))
        # Retain the original column-degree, left-multiplied normalization.
        graph = inverse[:,None]*graph
        support = torch.eye(self.nodes)
        supports = []
        for _ in range(hops):
            support = support@graph
            if clamp_adjacency:
                support = support.clamp_max(1.)
            supports.append(support)
        self.register_buffer('supports',torch.stack(supports))
        self.register_buffer('cell_filter',supports[-1].repeat(self.outputs,self.outputs))
        self.gc_list = nn.ModuleList(FilterLinear(matrix,self.features,self.outputs) for matrix in supports)
        width = self.nodes*self.outputs
        self.fl,self.il,self.ol,self.Cl = (nn.Linear(width*(hops+1),width) for _ in range(4))
        self.rfl,self.ril,self.rol,self.rCl = (nn.Linear(self.nodes*self.features+width,width) for _ in range(4))
        self.Neighbor_weight = nn.Parameter(torch.empty(width))
        nn.init.uniform_(self.Neighbor_weight,-1/math.sqrt(width),1/math.sqrt(width))
        # c stores an unconstrained coordinate; softplus(c) is the physical,
        # positive fusion coefficient and starts at the original value one.
        self.c = nn.Parameter(torch.tensor([math.log(math.expm1(1.))]))

    def one_step(self,source):
        width = self.nodes*self.outputs
        hidden = source.new_zeros(len(source),width)
        cell,self_hidden,self_cell = torch.zeros_like(hidden),torch.zeros_like(hidden),torch.zeros_like(hidden)
        neighbor_weight = self.cell_filter@self.Neighbor_weight
        coefficient = F.softplus(self.c).clamp_min(torch.finfo(source.dtype).tiny)
        for index in range(source.shape[1]):
            value = source[:,index].transpose(1,2).flatten(1)
            filtered = torch.cat([layer(value) for layer in self.gc_list],1)
            combined = torch.cat((filtered,hidden),1)
            cell = torch.sigmoid(self.fl(combined))*(cell*neighbor_weight) \
                +torch.sigmoid(self.il(combined))*torch.tanh(self.Cl(combined))
            hidden = torch.sigmoid(self.ol(combined))*torch.tanh(cell)
            combined = torch.cat((value,self_hidden),1)
            self_cell = torch.sigmoid(self.rfl(combined))*self_cell \
                +torch.sigmoid(self.ril(combined))*torch.tanh(self.rCl(combined))
            self_hidden = torch.sigmoid(self.rol(combined))*torch.tanh(self_cell)
            prediction = fuse_streams(hidden,self_hidden,value,filtered,coefficient)
        return prediction.reshape(len(source),self.outputs,self.nodes).transpose(1,2)

    def forward(self,batch):
        source = batch['X']
        if source.ndim!=4 or source.shape[1:]!=(self.length,self.nodes,self.features):
            raise ValueError('DKFN input must have declared [batch,time,node,feature] axes')
        predictions = []
        for index in range(self.horizon):
            prediction = self.one_step(source)
            predictions.append(prediction)
            if index+1<self.horizon:
                following = source[:,-1].clone()
                following[...,:self.outputs] = prediction
                if self.future_indices:
                    if 'future_features' not in batch:
                        raise ValueError('DKFN multistep rolling requires declared known future_features')
                    following[...,list(self.future_indices)] = batch['future_features'][:,index]
                source = torch.cat((source[:,1:],following[:,None]),1)
        return torch.stack(predictions,1)
