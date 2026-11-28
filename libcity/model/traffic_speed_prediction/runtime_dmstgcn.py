"""Dynamic primary/auxiliary temporal graphs with shape-derived forecast heads."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_norm import SingletonBatchNorm2d


class Pointwise(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.mlp = nn.Conv2d(inputs,outputs,1)

    def forward(self,values):
        return self.mlp(values)


class GraphConv(nn.Module):
    def __init__(self,inputs,outputs,dropout,order):
        super().__init__()
        self.mlp = Pointwise((order+1)*inputs,outputs)
        self.dropout,self.order = dropout,order

    def forward(self,values,graph):
        terms = [values]
        for _ in range(self.order):
            terms.append(torch.einsum('bcvt,bwv->bcwt',terms[-1],graph))
        return F.dropout(self.mlp(torch.cat(terms,1)),self.dropout,self.training)


def dynamic_graph(time,source,target,core):
    value = torch.einsum('bi,ijk->bjk',time,core)
    value = torch.einsum('nj,bjk->bnk',source,value)
    logits = torch.einsum('mk,bnk->bnm',target,value).relu()
    logits = logits.float() if logits.dtype in (torch.float16,torch.bfloat16) else logits
    return logits.softmax(-1)


class DMSTGCN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,num_layers=2,num_blocks=4,
                 residual_channels=32,dilation_channels=32,end_channels=512,kernel_size=2,
                 embedding_dims=40,order=2,dropout=.3,normalization='batch',skip_channels=8):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,num_layers=num_layers,
                num_blocks=num_blocks,residual_channels=residual_channels,dilation_channels=dilation_channels,
                end_channels=end_channels,kernel_size=kernel_size,embedding_dims=embedding_dims,
                skip_channels=skip_channels,num_nodes=metadata.num_nodes,output_dim=metadata.output_dim).items():
            if type(value) is not int or value<1:
                raise ValueError(f'DMSTGCN {name} must be a positive integer')
        if type(order) is not int or order<0 or type(dropout) not in (int,float) or not math.isfinite(dropout) or not 0<=dropout<1:
            raise ValueError('DMSTGCN requires nonnegative graph order and dropout in [0,1)')
        calendar = metadata.input_layout.get('history_features',{})
        streams = metadata.input_layout.get('dmstgcn_features',{})
        if metadata.input_layout.get('kind')!='window' or calendar.get('kind')!='calendar_positions' or not streams:
            raise ValueError('DMSTGCN requires observed streams and integer calendar history_features')
        if normalization not in ('batch','layer'):
            raise ValueError('DMSTGCN normalization must be batch or layer')
        self.nodes,self.features,self.outputs = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.input_window,self.output_window = input_window,output_window
        self.primary,self.auxiliary = streams['primary_index'],streams['auxiliary_index']
        self.slots = calendar['steps_per_day']
        observed = metadata.input_layout.get('observed_feature_indices',range(self.features))
        if type(self.slots) is not int or self.slots<1 or self.primary==self.auxiliary or any(
                type(index) is not int or index not in observed for index in (self.primary,self.auxiliary)):
            raise ValueError('DMSTGCN requires valid observed stream indices and positive steps_per_day')
        self.count = num_layers*num_blocks
        self.receptive_field = 1+num_blocks*(kernel_size-1)*(2**num_layers-1)
        length = max(input_window,self.receptive_field)
        self.lengths = []
        self.start_conv = nn.Conv2d(1,residual_channels,1)
        if self.count>1:
            self.start_conv_a = nn.Conv2d(1,residual_channels,1)
        for name in ('filter_convs','gate_convs','skip_convs','normal','gconv','filter_convs_a','gate_convs_a',
                'normal_a','gconv_a','gconv_a2p'):
            setattr(self,name,nn.ModuleList())
        if self.count>1 and order:
            for name in ('p','a','a2p'):
                for suffix,shape in (('1',(self.slots,embedding_dims)),('2',(self.nodes,embedding_dims)),
                        ('3',(self.nodes,embedding_dims)),('k',(embedding_dims,)*3)):
                    self.register_parameter('nodevec_'+name+suffix,nn.Parameter(torch.randn(*shape)))
        for index in range(self.count):
            dilation = 2**(index%num_layers)
            length -= dilation*(kernel_size-1)
            self.lengths.append(length)
            for name in ('filter_convs','gate_convs'):
                getattr(self,name).append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dilation)))
            self.skip_convs.append(nn.Conv2d(dilation_channels,skip_channels,1))
            if index+1==self.count:
                continue
            for name in ('filter_convs_a','gate_convs_a'):
                getattr(self,name).append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dilation)))
            self.gconv.append(GraphConv(dilation_channels,residual_channels,dropout,order))
            self.gconv_a.append(GraphConv(dilation_channels,residual_channels,dropout,order))
            self.gconv_a2p.append(GraphConv(residual_channels,residual_channels,dropout,order))
            self.normal.append(SingletonBatchNorm2d(residual_channels) if normalization=='batch' else
                nn.LayerNorm((residual_channels,self.nodes,length)))
            if index+2<self.count:
                self.normal_a.append(SingletonBatchNorm2d(residual_channels) if normalization=='batch' else
                    nn.LayerNorm((residual_channels,self.nodes,length)))
        self.end_conv_1 = nn.Conv2d(skip_channels*sum(self.lengths),end_channels,1)
        self.end_conv_2 = nn.Conv2d(end_channels,output_window*self.outputs,1)
        self.order = order

    def graphs(self,slots):
        if self.count==1 or not self.order:
            return (None,None,None)
        return tuple(dynamic_graph(getattr(self,'nodevec_'+name+'1')[slots],getattr(self,'nodevec_'+name+'2'),
            getattr(self,'nodevec_'+name+'3'),getattr(self,'nodevec_'+name+'k')) for name in ('p','a','a2p'))

    def forward(self,inputs):
        source,positions = inputs['X'],inputs.get('history_features')
        if source.ndim!=4 or source.shape[1:]!=(self.input_window,self.nodes,self.features):
            raise ValueError('DMSTGCN X must match configured [batch,history,node,feature] dimensions')
        if positions is None or positions.dtype!=torch.long or positions.shape!=(len(source),self.input_window,2):
            raise ValueError('DMSTGCN history_features must be int64 [batch,history,2] calendar positions')
        if torch.any((positions[...,1]<0)|(positions[...,1]>=self.slots)):
            raise ValueError('DMSTGCN calendar slots are outside steps_per_day')
        values = source.permute(0,3,2,1)
        values = F.pad(values,(max(0,self.receptive_field-self.input_window),0))
        primary = self.start_conv(values[:,self.primary:self.primary+1])
        auxiliary = self.start_conv_a(values[:,self.auxiliary:self.auxiliary+1]) if self.count>1 else None
        primary_graph,auxiliary_graph,transfer_graph = self.graphs(positions[:,0,1])
        skips = []
        for index in range(self.count):
            residual = primary
            primary = self.filter_convs[index](residual).tanh()*self.gate_convs[index](residual).sigmoid()
            skip = self.skip_convs[index](primary)
            skips.append(skip.transpose(2,3).flatten(1,2).unsqueeze(-1))
            if index+1==self.count:
                break
            residual_aux = auxiliary
            auxiliary = self.filter_convs_a[index](residual_aux).tanh()*self.gate_convs_a[index](residual_aux).sigmoid()
            primary = self.gconv[index](primary,primary_graph)
            auxiliary = self.gconv_a[index](auxiliary,auxiliary_graph)
            primary = primary+self.gconv_a2p[index](auxiliary,transfer_graph)
            primary = self.normal[index](primary+residual[...,-primary.shape[-1]:])
            if index+2<self.count:
                auxiliary = self.normal_a[index](auxiliary+residual_aux[...,-auxiliary.shape[-1]:])
        value = torch.cat(list(reversed(skips)),1).relu()
        value = self.end_conv_2(self.end_conv_1(value).relu()).squeeze(-1)
        return value.reshape(len(source),self.output_window,self.outputs,self.nodes).transpose(2,3)
