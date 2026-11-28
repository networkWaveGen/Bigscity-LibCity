"""Dynamic graph convolutions over explicitly ordered CPT history chunks."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_norm import SingletonBatchNorm1d,SingletonBatchNorm2d


def scaled_laplacian(adjacency):
    graph = np.array(adjacency,dtype=np.float64,copy=True)
    if graph.ndim!=2 or not len(graph) or graph.shape[0]!=graph.shape[1] or not np.isfinite(graph).all() or np.any(graph<0):
        raise ValueError('DGCN requires a finite nonnegative square adjacency')
    laplacian = np.diag(graph.sum(1))-graph
    maximum = float(np.linalg.eigvals(laplacian).real.max())
    scaled = 2*laplacian/maximum-np.eye(len(graph)) if maximum>1e-12 else -np.eye(len(graph))
    return torch.tensor(scaled,dtype=torch.float32)


def graph_support(base, residual, epsilon):
    value = base+residual
    denominator = value.sum(-1,keepdim=True)+epsilon
    minimum = epsilon*1e-4
    sign = torch.where(denominator<0,-torch.ones_like(denominator),torch.ones_like(denominator))
    denominator = torch.where(denominator.abs()<minimum,sign*minimum,denominator)
    return value/denominator


class SpatialAttention(nn.Module):
    def __init__(self, channels, heads, chunk=1):
        super().__init__()
        self.channels,self.heads,self.chunk = channels,heads,chunk
        self.conv1 = nn.Conv2d(channels*chunk,channels,1,bias=False)
        self.conv2 = nn.Conv2d(channels*chunk,channels,1,bias=False)

    def forward(self, value):
        batch,channels,nodes,length = value.shape
        if self.chunk>1:
            # One channel group contains the complete H-frame history chunk;
            # the last axis indexes chunks rather than interleaved time slots.
            value = value.reshape(batch,channels,nodes,length//self.chunk,self.chunk)
            value = value.permute(0,1,4,2,3).reshape(batch,channels*self.chunk,nodes,length//self.chunk)
        steps = value.shape[-1]
        left = self.conv1(value).reshape(batch,channels//self.heads,self.heads,nodes,steps).permute(0,3,1,4,2)
        right = self.conv2(value).reshape(batch,channels//self.heads,self.heads,nodes,steps).permute(0,1,3,4,2)
        score = torch.einsum('bnclh,bcmlh->bnmlh',left,right).permute(0,3,1,2,4)
        return score.sigmoid().mean(-1)


class TemporalAttention(nn.Module):
    def __init__(self, channels, nodes, groups):
        super().__init__()
        length = sum(groups)
        self.conv1 = nn.Conv2d(channels,1,1,bias=False)
        self.conv2 = nn.Conv2d(nodes,1,1,bias=False)
        self.w = nn.Parameter(torch.empty(nodes,channels))
        self.b = nn.Parameter(torch.zeros(length,length))
        self.v = nn.Parameter(torch.empty(length,length))
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.v)
        self.bn = SingletonBatchNorm1d(length)
        mask = torch.zeros(length,length,dtype=torch.bool)
        start = 0
        for width in groups:
            mask[start:start+width,start:start+width] = True
            start += width
        self.register_buffer('mask',mask)

    def forward(self, value):
        left = self.conv1(value.permute(0,1,3,2))[:,0]
        right = self.conv2(value.permute(0,2,1,3))[:,0]
        logits = self.v@torch.sigmoid(left@self.w@right+self.b)
        logits = self.bn(logits.transpose(1,2)).transpose(1,2)
        return logits.masked_fill(~self.mask,float('-inf')).softmax(-1)


class DynamicChebyshev(nn.Module):
    def __init__(self, inputs, outputs, nodes, order, kernel):
        super().__init__()
        self.order = order
        self.register_buffer('identity',torch.eye(nodes))
        self.conv1 = nn.Conv2d(order*inputs,outputs,(1,kernel),padding=(0,kernel//2))

    def forward(self, value, adjacency):
        terms = [self.identity.expand(value.shape[0],-1,-1)]
        if self.order>1:
            terms.append(adjacency)
        for _ in range(2,self.order):
            terms.append(2*adjacency@terms[-1]-terms[-2])
        basis = torch.stack(terms,1).transpose(-1,-2)
        features = torch.einsum('bcnl,bknq->bckql',value,basis).flatten(1,2)
        return self.conv1(features)


class DynamicBlock(nn.Module):
    def __init__(self, inputs, channels, nodes, horizon, chunks, groups, order, kernel, heads, dropout):
        super().__init__()
        self.horizon,self.chunks,self.dropout = horizon,chunks,dropout
        self.conv1 = nn.Conv2d(inputs,channels,1)
        self.time_conv = nn.Conv2d(inputs,channels,(1,kernel),padding=(0,kernel//2))
        if order>1:
            self.coarse_attention = SpatialAttention(channels,heads,horizon) if chunks>1 else None
            self.fine_attention = SpatialAttention(channels,heads)
            self.graph_recurrence = nn.LSTM(nodes,nodes,batch_first=True)
        else:
            self.coarse_attention = self.fine_attention = self.graph_recurrence = None
        self.graph_conv = DynamicChebyshev(channels,2*channels,nodes,order,kernel)
        self.temporal_attention = TemporalAttention(channels,nodes,groups)
        self.normalization = nn.LayerNorm((channels,nodes,horizon*chunks))

    def forward(self, source, support):
        value = F.leaky_relu(self.time_conv(source))
        adjacency = None
        if self.graph_recurrence is not None:
            matrices = []
            if self.coarse_attention is not None:
                matrices.append(self.coarse_attention(value[...,:-self.horizon]))
            matrices.append(self.fine_attention(value[...,-self.horizon:]))
            relations = torch.cat(matrices,1)
            batch,steps,nodes,_ = relations.shape
            sequence = relations.permute(0,2,1,3).reshape(batch*nodes,steps,nodes)
            _,(hidden,_) = self.graph_recurrence(F.dropout(sequence,self.dropout,self.training))
            adjacency = hidden[-1].reshape(batch,nodes,nodes)*support
        value = self.graph_conv(F.dropout(value,self.dropout,self.training),adjacency)
        signal,gate = value.chunk(2,1)
        value = F.dropout(gate.sigmoid()*F.leaky_relu(signal),self.dropout,self.training)
        temporal = self.temporal_attention(value).transpose(-1,-2)
        value = torch.einsum('bcnl,blq->bcnq',value,temporal)
        return self.normalization(F.leaky_relu(value)+self.conv1(source))


class DGCN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, c_out=64, K=3, Kt=3,
                 attention_heads=4, dropout=.5, normalization_epsilon=1e-4):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,c_out=c_out,K=K,Kt=Kt,
                attention_heads=attention_heads,num_nodes=metadata.num_nodes,feature_dim=metadata.feature_dim,
                output_dim=metadata.output_dim).items():
            if type(value) is not int or value<1:
                raise ValueError(f'DGCN {name} must be a positive integer')
        if Kt%2!=1 or K>1 and c_out%attention_heads:
            raise ValueError('DGCN requires an odd Kt and attention_heads dividing c_out when K > 1')
        if type(dropout) not in (int,float) or not math.isfinite(dropout) or not 0<=dropout<1:
            raise ValueError('DGCN dropout must be in [0,1)')
        if type(normalization_epsilon) not in (int,float) or not math.isfinite(normalization_epsilon) or normalization_epsilon<=0:
            raise ValueError('DGCN normalization_epsilon must be positive and finite')
        layout = metadata.input_layout
        if layout.get('kind')!='cpt':
            raise ValueError('DGCN requires explicit CPT sampling')
        sampling = layout['sampling']
        self.segments = {}
        start = 0
        for kind in ('trend','period','closeness'):
            count = sampling.get(kind,0)
            self.segments[kind] = (start,start+count*output_window,count)
            start += count*output_window
        if start!=layout['length'] or not start:
            raise ValueError('DGCN CPT lengths disagree with metadata')
        self.length,self.horizon = start,output_window
        self.nodes,self.features = metadata.num_nodes,metadata.feature_dim
        self.dropout,self.epsilon = dropout,normalization_epsilon
        chunks = start//output_window
        self.groups = [output_window]*(chunks-2)+[2*output_window] if chunks>=2 else [output_window]
        self.input_normalization = SingletonBatchNorm2d(metadata.feature_dim,affine=False)
        if K>1:
            support = scaled_laplacian(metadata.adjacency)
            if support.shape!=(self.nodes,self.nodes):
                raise ValueError('DGCN adjacency must match num_nodes')
            self.register_buffer('supports',support)
            self.graph_residual = nn.Parameter(torch.empty(self.nodes,self.nodes))
            nn.init.uniform_(self.graph_residual,0,.0001)
        else:
            self.register_buffer('supports',None)
            self.register_parameter('graph_residual',None)
        self.blocks = nn.ModuleList(DynamicBlock(inputs,c_out,self.nodes,output_window,chunks,self.groups,
            K,Kt,attention_heads,dropout) for inputs in (metadata.feature_dim,c_out))
        self.prediction_heads = nn.ModuleList(nn.Conv2d(c_out,metadata.output_dim,(1,width//output_window),
            stride=(1,width//output_window)) for width in self.groups)

    def ordered_history(self, source):
        values = []
        for kind in ('closeness','period','trend'):
            begin,end,count = self.segments[kind]
            if count:
                value = source[:,begin:end].reshape(len(source),count,self.horizon,self.nodes,self.features)
                value = value.flip(1).flatten(1,2).permute(0,3,2,1)
                values.append(self.input_normalization(value))
        return torch.cat(values,-1)

    def forward(self, inputs):
        source = inputs['X']
        if source.ndim!=4 or source.shape[1:]!=(self.length,self.nodes,self.features):
            raise ValueError('DGCN X must have configured [batch,CPT history,node,feature] dimensions')
        value = self.ordered_history(source)
        support = None if self.supports is None else F.dropout(graph_support(self.supports,self.graph_residual,self.epsilon),self.dropout,self.training)
        for block in self.blocks:
            value = block(value,support)
        outputs = []
        start = 0
        for head,width in zip(self.prediction_heads,self.groups):
            outputs.append(head(value[...,start:start+width]))
            start += width
        return torch.stack(outputs).sum(0).permute(0,3,2,1)
