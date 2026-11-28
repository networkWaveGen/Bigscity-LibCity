"""Periodic point forecasts with explicit graph/time axes and immutable supports."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(inputs,outputs))
        self.bias = nn.Parameter(torch.empty(1,outputs,1,1))
        bound = 1/math.sqrt(inputs)
        nn.init.uniform_(self.weight,-bound,bound)
        nn.init.uniform_(self.bias,-bound,bound)

    def forward(self,source,graph):
        support = torch.einsum('btnf,th->bhnf',source,self.weight)
        return torch.einsum('mn,bhnf->bhmf',graph,support)+self.bias


class GCN(nn.Module):
    def __init__(self,inputs,hidden,outputs):
        super().__init__()
        self.gc1,self.gc2 = GraphConvolution(inputs,hidden),GraphConvolution(hidden,outputs)

    def forward(self,source,graph):
        return self.gc2(F.relu(self.gc1(source,graph)),graph)


class CONVGCN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,conv_depth=5,conv_height=3,hidden_size=16):
        super().__init__()
        if metadata.input_layout.get('kind')!='seasonal_windows':
            raise ValueError('CONVGCN requires seasonal_windows sampling')
        if any(type(value) is not int or value<1 for value in (conv_depth,conv_height,hidden_size)):
            raise ValueError('CONVGCN dimensions must be positive integers')
        graph = np.array(metadata.adjacency,dtype=np.float32,copy=True)
        if graph.shape!=(metadata.num_nodes,metadata.num_nodes) or not np.isfinite(graph).all() or np.any(graph<0):
            raise ValueError('CONVGCN adjacency must be finite, nonnegative and match nodes')
        np.fill_diagonal(graph,1.)
        self.register_buffer('adj_mx',torch.tensor(graph))
        self.nodes,self.features,self.outputs = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.horizon,self.depth,self.height = output_window,conv_depth,conv_height
        self.length = metadata.input_layout['length']
        if self.length!=3*input_window:
            raise ValueError('CONVGCN requires three equal-length seasonal windows')
        self.gc = GCN(self.length,hidden_size,conv_depth*conv_height)
        self.Conv = nn.Conv3d(self.nodes,self.nodes,3,padding=1)
        self.fc = nn.Linear(self.nodes*conv_depth*conv_height*self.features,self.nodes*output_window*self.outputs)

    def forward(self,batch):
        source = batch['X']
        if source.ndim!=4 or source.shape[1:]!=(self.length,self.nodes,self.features):
            raise ValueError('CONVGCN input must have declared [batch,time,node,feature] axes')
        projected = self.gc(source,self.adj_mx)
        # Node is the Conv3d channel, not a factor of the graph time projection.
        packed = projected.permute(0,2,1,3).reshape(-1,self.nodes,self.depth,self.height,self.features)
        prediction = self.fc(F.relu(self.Conv(packed)).flatten(1))
        return prediction.reshape(-1,self.horizon,self.nodes,self.outputs)


class TemporalBlock(nn.Module):
    def __init__(self,nodes,features):
        super().__init__()
        width = nodes*features
        self.lstm,self.lstm2 = nn.LSTM(width,width,1,batch_first=True),nn.LSTM(width,width,1,batch_first=True)
        self.linear = nn.Linear(width,width)

    def forward(self,source):
        hidden,_ = self.lstm(source.flatten(2))
        hidden,_ = self.lstm2(hidden)
        return F.relu(self.linear(hidden[:,-1]))


class SpatialBlock(nn.Module):
    def __init__(self,graph,nodes,features):
        super().__init__()
        self.register_buffer('S',torch.tensor(np.array(graph,copy=True),dtype=torch.float32))
        width = nodes*features
        self.linear1,self.linear2,self.linear3 = (nn.Linear(width,width) for _ in range(3))
        self.lstm,self.lstm2 = nn.LSTM(width,width,1,batch_first=True),nn.LSTM(width,width,1,batch_first=True)

    def forward(self,source):
        propagated = self.S@source
        hidden = F.relu(self.linear1(propagated.flatten(2))).reshape_as(source)
        hidden = F.relu(self.linear2((self.S@hidden).flatten(2)))
        hidden,_ = self.lstm(hidden)
        hidden,_ = self.lstm2(hidden)
        return F.relu(self.linear3(hidden[:,-1]))


class SpatialComponent(nn.Module):
    def __init__(self,graphs,nodes,features,outputs):
        super().__init__()
        for name,graph in zip(('near_block','middle_block','distant_block'),graphs):
            setattr(self,name,SpatialBlock(graph,nodes,features))
        self.linear = nn.Linear(3*nodes*features,outputs)

    def forward(self,source):
        pieces = [getattr(self,name)(source) for name in ('near_block','middle_block','distant_block')]
        return F.relu(self.linear(torch.cat(pieces,1)))


class TemporalComponent(nn.Module):
    def __init__(self,nodes,features,outputs,counts):
        super().__init__()
        self.segments = []
        start = 0
        # Recent/interval, daily and weekly have distinct learned recurrent blocks.
        for name,count in zip(('interval_block','daily_block','weekly_block'),counts):
            if count:
                setattr(self,name,TemporalBlock(nodes,features))
                self.segments.append((name,start,start+count))
            start += count
        self.linear = nn.Linear(len(self.segments)*nodes*features,outputs)

    def forward(self,source):
        pieces = [getattr(self,name)(source[:,start:end]) for name,start,end in self.segments]
        return F.relu(self.linear(torch.cat(pieces,1)))


class MultiSTGCnet(nn.Module):
    def __init__(self,*,metadata,input_window,output_window):
        super().__init__()
        if metadata.input_layout.get('kind')!='frame_cpt' or 'distance_bands' not in metadata.input_layout:
            raise ValueError('MultiSTGCnet requires frame_cpt sampling and explicit distance bands')
        sampling = metadata.input_layout['sampling']
        counts = tuple(sampling.get(name,0) for name in ('closeness','period','trend'))
        graphs = np.asarray(metadata.input_layout['distance_bands'])
        nodes,features = metadata.num_nodes,metadata.feature_dim
        if graphs.shape!=(3,nodes,nodes) or not np.isfinite(graphs).all() or np.any((graphs!=0)&(graphs!=1)):
            raise ValueError('MultiSTGCnet needs three finite binary node-aligned distance bands')
        self.nodes,self.outputs,self.horizon = nodes,metadata.output_dim,output_window
        self.length,self.features,self.closeness = sum(counts),features,counts[0]
        width = output_window*nodes*metadata.output_dim
        if self.closeness:
            self.spatial_component = SpatialComponent(graphs,nodes,features,width)
            self.Ws = nn.Parameter(torch.empty(1,width))
            nn.init.normal_(self.Ws,std=.01)
        self.temporal_component = TemporalComponent(nodes,features,width,counts)
        self.Wt = nn.Parameter(torch.empty(1,width))
        nn.init.normal_(self.Wt,std=.01)

    def forward(self,batch):
        source = batch['X']
        if source.ndim!=4 or source.shape[1:]!=(self.length,self.nodes,self.features):
            raise ValueError('MultiSTGCnet input must have declared [batch,CPT time,node,feature] axes')
        prediction = self.Wt*self.temporal_component(source)
        if self.closeness:
            prediction = prediction+self.Ws*self.spatial_component(source[:,:self.closeness])
        return prediction.reshape(-1,self.horizon,self.nodes,self.outputs)
