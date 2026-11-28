"""Node/region graph convolutions and learned masked cross-level transfer."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_attention import TemporalAttention
from libcity.model.runtime_norm import SingletonBatchNorm2d


from libcity.model.runtime_graph import normalized_supports


def graph_supports(adjacency,kind):
    try:
        return normalized_supports(adjacency,kind)
    except ValueError as error:
        raise ValueError('HGCN '+str(error)) from error


def adaptive_support(left, right, identity):
    score = left@right
    score = score.float() if score.dtype in (torch.float16,torch.bfloat16) else score
    value = score.relu()
    degree = value.sum(-1,keepdim=True)
    normalized = value/torch.where(degree>0,degree,torch.ones_like(degree))
    return torch.where(degree>0,normalized,identity)


class GraphTimeConvolution(nn.Module):
    def __init__(self, inputs, outputs, kernel, support_count, order, dropout):
        super().__init__()
        self.order,self.dropout = order,dropout
        self.mlp = nn.Conv2d(inputs*(1+order*support_count),outputs,(1,kernel))

    def forward(self, value, supports):
        features = [value]
        for adjacency in supports:
            propagated = value
            for _ in range(self.order):
                propagated = torch.einsum('nm,bcmt->bcnt',adjacency,propagated)
                features.append(propagated)
        return F.dropout(self.mlp(torch.cat(features,1)),self.dropout,self.training)


class GraphPool(nn.Module):
    def __init__(self, inputs, channels, nodes, steps, kernel, supports, order, dropout):
        super().__init__()
        self.time_conv = nn.Conv2d(inputs,2*channels,(1,kernel),dilation=2)
        self.graph_conv = GraphTimeConvolution(channels,2*channels,kernel,supports,order,dropout)
        self.temporal_attention = TemporalAttention(channels,nodes,[steps])
        self.normalization = SingletonBatchNorm2d(channels)
        self.residual = nn.Conv2d(inputs,channels,1)

    def forward(self, value, supports):
        residual = self.residual(value)
        signal,gate = self.time_conv(value).chunk(2,1)
        signal,gate = self.graph_conv(signal.tanh()*gate.sigmoid(),supports).chunk(2,1)
        value = signal.tanh()*gate.sigmoid()
        temporal = self.temporal_attention(value).transpose(-1,-2)
        value = torch.einsum('bcnl,blq->bcnq',value,temporal)
        return self.normalization(value+residual[...,-value.shape[-1]:])


class Transmit(nn.Module):
    def __init__(self, channels, steps, nodes, clusters):
        super().__init__()
        self.conv1 = nn.Conv2d(channels,1,1,bias=False)
        self.conv2 = nn.Conv2d(steps,1,1,bias=False)
        self.w = nn.Parameter(torch.empty(steps,channels))
        self.b = nn.Parameter(torch.zeros(nodes,clusters))
        nn.init.xavier_uniform_(self.w)

    def forward(self, nodes, clusters, membership):
        left = self.conv1(nodes)[:,0]
        right = self.conv2(clusters.permute(0,3,1,2))[:,0]
        value = torch.sigmoid(left@self.w@right+self.b)
        return torch.sigmoid(value-value.mean(1,keepdim=True))*membership


class HGCN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, channels=32, skip_channels=256,
                 end_channels=512, adjtype='doubletransition', dropout=.3, adaptive_dim=10,
                 graph_order=2, temporal_kernels=(3,2)):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,channels=channels,
                skip_channels=skip_channels,end_channels=end_channels,adaptive_dim=adaptive_dim,graph_order=graph_order,
                num_nodes=metadata.num_nodes,feature_dim=metadata.feature_dim,output_dim=metadata.output_dim).items():
            if type(value) is not int or value<1:
                raise ValueError(f'HGCN {name} must be a positive integer')
        if not isinstance(temporal_kernels,(tuple,list)) or len(temporal_kernels)!=2 or any(type(k) is not int or k<1 for k in temporal_kernels):
            raise ValueError('HGCN temporal_kernels requires two positive integer kernels')
        stages = [input_window-3*(temporal_kernels[0]-1)]
        stages.append(stages[-1]-3*(temporal_kernels[1]-1))
        if min(stages)<1:
            raise ValueError('HGCN input_window is too short for the configured temporal_kernels')
        if type(dropout) not in (int,float) or not math.isfinite(dropout) or not 0<=dropout<1:
            raise ValueError('HGCN dropout must be in [0,1)')
        if metadata.input_layout.get('kind')!='window' or 'hierarchy' not in metadata.input_layout:
            raise ValueError('HGCN requires static hierarchical window metadata')
        hierarchy = metadata.input_layout['hierarchy']
        membership = np.array(hierarchy['membership'],dtype=np.float32,copy=True)
        if (membership.ndim!=2 or membership.shape[0]!=metadata.num_nodes or membership.shape[1]<1
                or not np.isin(membership,(0.,1.)).all() or not np.all(membership.sum(1)==1)
                or np.any(membership.sum(0)==0)):
            raise ValueError('HGCN membership must be a nonempty one-hot partition of every node')
        clusters = membership.shape[1]
        self.nodes,self.features,self.input_window = metadata.num_nodes,metadata.feature_dim,input_window
        self.register_buffer('membership',torch.from_numpy(membership))
        self.register_buffer('supports',graph_supports(metadata.adjacency,adjtype))
        self.register_buffer('supports_cluster',graph_supports(hierarchy['cluster_adjacency'],adjtype))
        if self.supports.shape[-1]!=self.nodes or self.supports_cluster.shape[-1]!=clusters:
            raise ValueError('HGCN graph dimensions must match node and cluster counts')
        self.register_buffer('node_identity',torch.eye(self.nodes))
        self.register_buffer('cluster_identity',torch.eye(clusters))
        self.nodevec1 = nn.Parameter(torch.randn(self.nodes,adaptive_dim))
        self.nodevec2 = nn.Parameter(torch.randn(adaptive_dim,self.nodes))
        self.nodevec1_c = nn.Parameter(torch.randn(clusters,adaptive_dim))
        self.nodevec2_c = nn.Parameter(torch.randn(adaptive_dim,clusters))
        self.bn = SingletonBatchNorm2d(self.features,affine=False)
        self.bn_cluster = SingletonBatchNorm2d(self.features,affine=False)
        self.start_conv = nn.Conv2d(self.features,channels,1)
        self.start_conv_cluster = nn.Conv2d(self.features,channels,1)
        self.node_blocks = nn.ModuleList(GraphPool(2*channels,channels,self.nodes,steps,kernel,
            len(self.supports)+1,graph_order,dropout) for steps,kernel in zip(stages,temporal_kernels))
        self.cluster_blocks = nn.ModuleList(GraphPool(channels,channels,clusters,steps,kernel,
            len(self.supports_cluster)+1,graph_order,dropout) for steps,kernel in zip(stages,temporal_kernels))
        self.transfers = nn.ModuleList(Transmit(channels,steps,self.nodes,clusters) for steps in (input_window,*stages))
        self.skips = nn.ModuleList(nn.Conv2d(2*channels,skip_channels,1) for _ in stages)
        self.end_conv_1 = nn.Conv2d(skip_channels,end_channels,(1,stages[-1]))
        self.linear = nn.Linear(1,metadata.output_dim)
        self.end_conv_2 = nn.Conv2d(end_channels,output_window,1)

    def cluster_inputs(self, value):
        return torch.einsum('bcnt,nk->bckt',value,self.membership)

    def forward(self, inputs):
        source = inputs['X']
        if source.ndim!=4 or source.shape[1:]!=(self.input_window,self.nodes,self.features):
            raise ValueError('HGCN X must have configured [batch,history,node,feature] dimensions')
        source = source.permute(0,3,2,1)
        clusters = self.start_conv_cluster(self.bn_cluster(self.cluster_inputs(source)))
        nodes = self.start_conv(self.bn(source))
        node_graph = adaptive_support(self.nodevec1,self.nodevec2,self.node_identity)
        cluster_graph = adaptive_support(self.nodevec1_c,self.nodevec2_c,self.cluster_identity)
        node_supports = torch.cat((self.supports,node_graph[None]),0)
        cluster_supports = torch.cat((self.supports_cluster,cluster_graph[None]),0)
        transferred = self.transfers[0](nodes,clusters,self.membership)
        nodes = torch.cat((nodes,torch.einsum('bnk,bckt->bcnt',transferred,clusters)),1)
        skip = None
        for index,(node_block,cluster_block) in enumerate(zip(self.node_blocks,self.cluster_blocks)):
            clusters = cluster_block(clusters,cluster_supports)
            nodes = node_block(nodes,node_supports)
            transferred = self.transfers[index+1](nodes,clusters,self.membership)
            nodes = torch.cat((nodes,torch.einsum('bnk,bckt->bcnt',transferred,clusters)),1)
            addition = self.skips[index](nodes)
            skip = addition if skip is None else skip[...,-addition.shape[-1]:]+addition
        value = F.relu(self.end_conv_1(F.relu(skip)))
        return self.end_conv_2(self.linear(value))
