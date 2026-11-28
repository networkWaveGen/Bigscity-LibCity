"""Residual flow/topology streams with sample-independent node-axis LSTMs."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_norm import SingletonBatchNorm2d


def normalized_adjacency(adjacency):
    graph = np.array(adjacency,dtype=np.float64,copy=True)
    if graph.ndim!=2 or graph.shape[0]!=graph.shape[1] or not len(graph) or not np.isfinite(graph).all() or np.any(graph<0):
        raise ValueError('ResLSTM adjacency must be finite, square and nonnegative')
    graph += np.eye(len(graph))
    inverse = graph.sum(0)**-.5
    return inverse[:,None]*graph*inverse[None,:]


class Unit(nn.Module):
    def __init__(self,inputs,outputs,pool=False):
        super().__init__()
        self.pool = pool
        if pool:
            self.conv1 = nn.Conv2d(inputs,outputs,1,stride=2)
        self.bn1,self.bn2 = SingletonBatchNorm2d(inputs),SingletonBatchNorm2d(outputs)
        self.conv2,self.conv3 = nn.Conv2d(inputs,outputs,3,padding=1),nn.Conv2d(outputs,outputs,3,padding=1)

    def forward(self,source):
        residual = self.conv1(source) if self.pool else source
        source = F.max_pool2d(source,2,2,ceil_mode=True) if self.pool else source
        hidden = self.conv2(F.relu(self.bn1(source)))
        return residual+self.conv3(F.relu(self.bn2(hidden)))


class ConvBlock(nn.Module):
    def __init__(self,channels,nodes,history,filters):
        super().__init__()
        self.conv_layer = nn.Conv2d(channels,filters,3,padding=1)
        self.unit1,self.unit2 = Unit(filters,filters),Unit(filters,filters*2,pool=True)
        self.fc = nn.Linear(2*filters*((nodes+1)//2)*((history+1)//2),nodes)

    def forward(self,source):
        return self.fc(self.unit2(self.unit1(self.conv_layer(source))).flatten(1))


class Attention3dBlock(nn.Module):
    def __init__(self,nodes):
        super().__init__()
        self.fc = nn.Linear(nodes,nodes)

    def forward(self,source):
        return source*self.fc(source.transpose(1,2)).transpose(1,2)


class ResLSTM(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,base_channels=32,recurrent_hidden=128,
                 recurrent_layers=2,flow_indices=(0,1)):
        super().__init__()
        layout = metadata.input_layout
        if layout.get('kind')!='seasonal_windows' or 'reslstm_observation_dim' not in layout:
            raise ValueError('ResLSTM requires seasonal_windows and explicit flow/external metadata')
        if any(type(value) is not int or value<1 for value in (base_channels,recurrent_hidden,recurrent_layers)):
            raise ValueError('ResLSTM channel and recurrent dimensions must be positive integers')
        if (not isinstance(flow_indices,(tuple,list)) or len(flow_indices)!=2
                or any(type(value) is not int or not 0<=value<layout['reslstm_observation_dim'] for value in flow_indices)
                or len(set(flow_indices))!=2):
            raise ValueError('ResLSTM flow_indices must select two distinct raw observation columns')
        self.flow_indices = tuple(flow_indices)
        self.nodes,self.features,self.outputs = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.history,self.horizon = input_window,output_window
        if layout['length']!=3*input_window:
            raise ValueError('ResLSTM needs three equal-length seasonal windows')
        support = normalized_adjacency(metadata.adjacency)
        if support.shape!=(self.nodes,self.nodes):
            raise ValueError('ResLSTM adjacency and node count differ')
        self.register_buffer('adj_mx',torch.tensor(support,dtype=torch.float32))
        self.conv_block1,self.conv_block2,self.conv_block3 = (
            ConvBlock(channels,self.nodes,input_window,base_channels) for channels in (3,3,1))
        self.external_indices = tuple(layout['reslstm_external_indices'])
        if self.external_indices:
            self.fc1 = nn.Linear(len(self.external_indices)*input_window,self.nodes)
            self.lstm1 = nn.LSTM(1,recurrent_hidden,recurrent_layers,batch_first=True)
            self.lstm2 = nn.LSTM(recurrent_hidden,1,recurrent_layers,batch_first=True)
            self.fc2 = nn.Linear(self.nodes,self.nodes)
        self.lstm3 = nn.LSTM(1,recurrent_hidden,recurrent_layers,batch_first=True)
        self.att = Attention3dBlock(self.nodes)
        self.fc_last = nn.Linear(self.nodes*recurrent_hidden,self.nodes*output_window*self.outputs)

    def packed_flows(self,source):
        # [trend, period, recent] are channels; frames are chronological within
        # each channel. A plain [node,history,3] view mixed these two axes.
        return source.reshape(len(source),3,self.history,self.nodes,self.features).permute(0,1,3,2,4)

    def fourth_pro(self,source):
        hidden = self.fc1(source.flatten(1)).unsqueeze(-1)
        hidden,_ = self.lstm1(hidden)
        hidden,_ = self.lstm2(hidden)
        return self.fc2(hidden.flatten(1))

    def forward(self,batch):
        source = batch['X']
        if source.ndim!=4 or source.shape[1:]!=(3*self.history,self.nodes,self.features):
            raise ValueError('ResLSTM input must have declared [batch,seasonal time,node,feature] axes')
        packed = self.packed_flows(source)
        recent = source[:,-self.history:,:,self.flow_indices[0]].transpose(1,2)
        graph = (self.adj_mx@recent).unsqueeze(1)
        hidden = self.conv_block1(packed[...,self.flow_indices[0]]) \
            +self.conv_block2(packed[...,self.flow_indices[1]])+self.conv_block3(graph)
        if self.external_indices:
            external = source[:,-self.history:,0,list(self.external_indices)].transpose(1,2)
            hidden = hidden+self.fourth_pro(external)
        hidden,_ = self.lstm3(hidden.unsqueeze(-1))
        prediction = self.fc_last(self.att(hidden).flatten(1))
        return prediction.reshape(-1,self.horizon,self.nodes,self.outputs)
