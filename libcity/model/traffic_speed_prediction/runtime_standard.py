"""Standard forecast modules use the shared loop without executor subclasses."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from libcity.model.traffic_speed_prediction.GWNET import GCN, asym_adj


class FNN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,hidden_size=128):
        super().__init__()
        self.output_dim,self.output_window=metadata.output_dim,output_window
        self.fc1=nn.Linear(input_window*metadata.feature_dim,hidden_size)
        self.fc2=nn.Linear(hidden_size,output_window*metadata.output_dim)
    def forward(self,batch):
        source=batch['X'].transpose(1,2).flatten(2)
        prediction=self.fc2(F.relu(self.fc1(source)))
        return prediction.reshape(source.shape[0],source.shape[1],self.output_window,self.output_dim).transpose(1,2)


class GWNET(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,nhid=32,blocks=4,layers=2,kernel_size=2,
                 dropout=.3,gcn_bool=True,addaptadj=True,aptonly=True):
        super().__init__()
        if min(blocks,layers,nhid)<1 or kernel_size<2 or metadata.output_dim!=1:
            raise ValueError('GWNET requires positive dimensions, kernel_size>=2 and scalar node targets')
        self.blocks,self.layers,self.gcn_bool,self.addaptadj=blocks,layers,gcn_bool,addaptadj
        self.filter_convs,self.gate_convs,self.residual_convs,self.skip_convs,self.bn,self.gconv=(nn.ModuleList() for _ in range(6))
        self.start_conv=nn.Conv2d(metadata.feature_dim,nhid,1)
        supports=[] if aptonly else [asym_adj(metadata.adjacency),asym_adj(metadata.adjacency.T)]
        self.support_count=len(supports)
        for index,graph in enumerate(supports):
            self.register_buffer(f'support_{index}',torch.tensor(np.asarray(graph),dtype=torch.float32))
        if addaptadj:
            self.nodevec1=nn.Parameter(torch.randn(metadata.num_nodes,10))
            self.nodevec2=nn.Parameter(torch.randn(10,metadata.num_nodes))
        support_count=len(supports)+int(addaptadj)
        self.receptive_field=1
        for _ in range(blocks):
            for index in range(layers):
                dilation=2**index
                self.filter_convs.append(nn.Conv2d(nhid,nhid,(1,kernel_size),dilation=(1,dilation)))
                self.gate_convs.append(nn.Conv2d(nhid,nhid,(1,kernel_size),dilation=(1,dilation)))
                self.residual_convs.append(nn.Conv2d(nhid,nhid,1))
                self.skip_convs.append(nn.Conv2d(nhid,nhid*8,1))
                self.bn.append(nn.BatchNorm2d(nhid))
                self.gconv.append(GCN(nhid,nhid,dropout,support_len=support_count))
                self.receptive_field+=(kernel_size-1)*dilation
        self.end_conv_1=nn.Conv2d(nhid*8,nhid*16,1)
        self.end_conv_2=nn.Conv2d(nhid*16,output_window,1)

    def forward(self,batch):
        source=F.pad(batch['X'].transpose(1,3),(1,0))
        source=F.pad(source,(max(0,self.receptive_field-source.shape[-1]),0))
        x=self.start_conv(source)
        skip=None
        supports=[getattr(self,f'support_{index}') for index in range(self.support_count)]
        if self.addaptadj:
            supports.append(torch.softmax(F.relu(self.nodevec1@self.nodevec2),dim=1))
        for index in range(self.blocks*self.layers):
            residual=x
            x=torch.tanh(self.filter_convs[index](x))*torch.sigmoid(self.gate_convs[index](x))
            contribution=self.skip_convs[index](x)
            skip=contribution if skip is None else skip[...,-contribution.shape[-1]:]+contribution
            x=self.gconv[index](x,supports) if self.gcn_bool and supports else self.residual_convs[index](x)
            x=self.bn[index](x+residual[...,-x.shape[-1]:])
        # One forecast per input window; long histories may leave several temporal positions.
        return self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))[...,-1:]
