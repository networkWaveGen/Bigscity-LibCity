"""MTGNN architecture; subgraph updates and curriculum state belong to its trainer."""
import torch
from torch import nn
from torch.nn import functional as F
from libcity.model.traffic_speed_prediction.MTGNN import MixProp,DilatedInception,LayerNorm


DEFAULTS = dict(gcn_true=True,buildA_true=True,gcn_depth=2,dropout=.3,subgraph_size=20,node_dim=40,
    dilation_exponential=1,conv_channels=32,residual_channels=32,skip_channels=64,end_channels=128,
    layers=3,propalpha=.05,tanhalpha=3.,layer_norm_affline=True)


class GraphConstructor(nn.Module):
    def __init__(self,nodes,k,dimension,alpha):
        super().__init__()
        self.emb1,self.emb2 = nn.Embedding(nodes,dimension),nn.Embedding(nodes,dimension)
        self.lin1,self.lin2 = nn.Linear(dimension,dimension),nn.Linear(dimension,dimension)
        self.k,self.alpha = k,alpha

    def forward(self,indices):
        first = torch.tanh(self.alpha*self.lin1(self.emb1(indices)))
        second = torch.tanh(self.alpha*self.lin2(self.emb2(indices)))
        relation = first@second.T-second@first.T
        adjacency = F.relu(torch.tanh(self.alpha*relation))
        values,selected = adjacency.topk(min(self.k,len(indices)),1)
        mask = torch.zeros_like(adjacency).scatter_(1,selected,torch.ones_like(values))
        return adjacency*mask


class MTGNN(nn.Module):
    def __init__(self, *,metadata,input_window,output_window,**parameters):
        super().__init__()
        if parameters.keys()-DEFAULTS.keys():
            raise ValueError('Unknown MTGNN architecture parameters')
        for key,value in {**DEFAULTS,**parameters}.items():
            setattr(self,key,value)
        self.num_nodes,self.feature_dim,self.output_dim = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        if self.output_dim!=1:
            raise ValueError('MTGNN convolution output supports output_dim=1')
        self.input_window,self.output_window = input_window,output_window
        self.register_buffer('idx',torch.arange(self.num_nodes))
        self.register_buffer('predefined_A',torch.tensor(metadata.adjacency.copy(),dtype=torch.float32)-torch.eye(self.num_nodes))
        self.filter_convs,self.gate_convs = nn.ModuleList(),nn.ModuleList()
        self.residual_convs,self.skip_convs = nn.ModuleList(),nn.ModuleList()
        self.gconv1,self.gconv2,self.norm = nn.ModuleList(),nn.ModuleList(),nn.ModuleList()
        self.start_conv = nn.Conv2d(self.feature_dim,self.residual_channels,(1,1))
        self.gc = GraphConstructor(self.num_nodes,self.subgraph_size,self.node_dim,self.tanhalpha)
        kernel = 7
        self.receptive_field = (int(self.output_dim+(kernel-1)*(self.dilation_exponential**self.layers-1)
            /(self.dilation_exponential-1)) if self.dilation_exponential>1 else self.layers*(kernel-1)+self.output_dim)
        dilation = 1
        for index in range(1,self.layers+1):
            width = (int(1+(kernel-1)*(self.dilation_exponential**index-1)/(self.dilation_exponential-1))
                     if self.dilation_exponential>1 else 1+index*(kernel-1))
            self.filter_convs.append(DilatedInception(self.residual_channels,self.conv_channels,dilation_factor=dilation))
            self.gate_convs.append(DilatedInception(self.residual_channels,self.conv_channels,dilation_factor=dilation))
            self.residual_convs.append(nn.Conv2d(self.conv_channels,self.residual_channels,(1,1)))
            self.skip_convs.append(nn.Conv2d(self.conv_channels,self.skip_channels,
                (1,max(self.input_window,self.receptive_field)-width+1)))
            if self.gcn_true:
                self.gconv1.append(MixProp(self.conv_channels,self.residual_channels,self.gcn_depth,self.dropout,self.propalpha))
                self.gconv2.append(MixProp(self.conv_channels,self.residual_channels,self.gcn_depth,self.dropout,self.propalpha))
            self.norm.append(LayerNorm((self.residual_channels,self.num_nodes,max(self.input_window,self.receptive_field)-width+1),
                                       elementwise_affine=self.layer_norm_affline))
            dilation *= self.dilation_exponential
        self.end_conv_1 = nn.Conv2d(self.skip_channels,self.end_channels,(1,1),bias=True)
        self.end_conv_2 = nn.Conv2d(self.end_channels,self.output_window,(1,1),bias=True)
        self.skip0 = nn.Conv2d(self.feature_dim,self.skip_channels,(1,max(self.input_window,self.receptive_field)),bias=True)
        self.skipE = nn.Conv2d(self.residual_channels,self.skip_channels,(1,max(1,self.input_window-self.receptive_field+1)),bias=True)

    def forward(self,batch):
        inputs = batch['X'].transpose(1,3)
        indices = batch.get('node_idx',self.idx)
        if inputs.shape[-1]!=self.input_window:
            raise ValueError('MTGNN history length differs from configured input_window')
        if self.input_window<self.receptive_field:
            inputs = F.pad(inputs,(self.receptive_field-self.input_window,0,0,0))
        if self.gcn_true:
            adjacency = self.gc(indices) if self.buildA_true else self.predefined_A[indices][:,indices]
        value = self.start_conv(inputs)
        skip = self.skip0(F.dropout(inputs,self.dropout,training=self.training))
        for index in range(self.layers):
            residual = value
            value = torch.tanh(self.filter_convs[index](value))*torch.sigmoid(self.gate_convs[index](value))
            value = F.dropout(value,self.dropout,training=self.training)
            skip = self.skip_convs[index](value)+skip
            value = (self.gconv1[index](value,adjacency)+self.gconv2[index](value,adjacency.T)
                     if self.gcn_true else self.residual_convs[index](value))
            value = self.norm[index](value+residual[:,:,:,-value.size(3):],indices)
        return self.end_conv_2(F.relu(self.end_conv_1(F.relu(self.skipE(value)+skip))))
