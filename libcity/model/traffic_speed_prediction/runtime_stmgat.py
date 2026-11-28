"""STMGAT with native directed edge attention and explicit temporal widths."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_norm import SingletonBatchNorm2d


def attention_edges(adjacency,nodes):
    graph=np.array(adjacency,dtype=np.float64,copy=True)
    if graph.shape!=(nodes,nodes) or not np.isfinite(graph).all() or np.any(graph<0):
        raise ValueError('STMGAT requires a finite nonnegative adjacency aligned to nodes')
    np.fill_diagonal(graph,0.)
    source,destination=np.nonzero(graph>0)
    # One self edge per node, including nodes with no incoming road edges.
    return torch.tensor(np.stack((np.r_[source,np.arange(nodes)],np.r_[destination,np.arange(nodes)])),dtype=torch.long)


class EdgeAttention(nn.Module):
    def __init__(self,width,heads,feat_drop,attn_drop,negative_slope):
        super().__init__();self.width,self.heads=width,heads;self.negative_slope=negative_slope
        self.fc=nn.Linear(width,heads*width,bias=False)
        self.attn_l=nn.Parameter(torch.empty(1,heads,width));self.attn_r=nn.Parameter(torch.empty(1,heads,width))
        self.bias=nn.Parameter(torch.zeros(heads*width))
        self.feat_drop=nn.Dropout(feat_drop);self.attn_drop=nn.Dropout(attn_drop)
        for value in (self.fc.weight,self.attn_l,self.attn_r):
            nn.init.xavier_normal_(value,gain=math.sqrt(2.))

    def forward(self,source,edge_index):
        projected=self.fc(self.feat_drop(source)).unflatten(-1,(self.heads,self.width))
        origin,destination=edge_index
        with torch.autocast(source.device.type,enabled=False):
            values=projected.float() if projected.dtype in (torch.float16,torch.bfloat16) else projected
            left=(values*self.attn_l).sum(-1);right=(values*self.attn_r).sum(-1)
            score=F.leaky_relu(left[:,origin]+right[:,destination],self.negative_slope)
            indices=destination[None,:,None].expand_as(score)
            maxima=score.new_full((*source.shape[:2],self.heads),-torch.inf)
            maxima.scatter_reduce_(1,indices,score.detach(),reduce='amax')
            exponential=(score-maxima[:,destination]).exp()
            denominator=score.new_zeros(maxima.shape).index_add(1,destination,exponential)
            attention=self.attn_drop(exponential/denominator[:,destination])
            weighted=values[:,origin]*attention.unsqueeze(-1)
            aggregate=weighted.new_zeros(values.shape).index_add(1,destination,weighted)
            output=F.elu(aggregate+self.bias.reshape(self.heads,self.width)).mean(2)
        return output.to(projected.dtype)


class STMGAT(nn.Module):
    def __init__(self,metadata,input_window,output_window,blocks=4,layers=2,run_gconv=True,
                 residual_channels=40,dilation_channels=40,skip_channels=320,end_channels=640,
                 kernel_size=2,heads=8,feat_drop=.6,attn_drop=.6,negative_slope=.2):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,blocks=blocks,layers=layers,
                residual_channels=residual_channels,dilation_channels=dilation_channels,skip_channels=skip_channels,
                end_channels=end_channels,kernel_size=kernel_size,heads=heads).items():
            if type(value) is not int or value<1: raise ValueError('STMGAT '+name+' must be a positive integer')
        if type(run_gconv) is not bool or any(not math.isfinite(value) or not 0<=value<1 for value in (feat_drop,attn_drop)) or not math.isfinite(negative_slope) or negative_slope<0:
            raise ValueError('STMGAT graph switch, dropout or negative slope is invalid')
        if run_gconv and blocks*layers>1 and residual_channels!=dilation_channels:
            raise ValueError('STMGAT graph residual requires equal residual_channels and dilation_channels')
        if metadata.input_layout.get('kind')!='window': raise ValueError('STMGAT requires continuous window metadata')
        self.steps,self.horizon,self.nodes,self.features,self.outputs=input_window,output_window,metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.run_gconv=run_gconv
        self.receptive_field=1+blocks*(kernel_size-1)*(2**layers-1)
        remaining=max(input_window,self.receptive_field)
        graph_used=run_gconv and blocks*layers>1
        self.register_buffer('edge_index',attention_edges(metadata.adjacency,self.nodes) if graph_used else torch.empty(2,0,dtype=torch.long))
        self.start_conv=nn.Conv2d(self.features,residual_channels,1)
        self.cat_feature_conv=nn.Conv2d(self.features,residual_channels,1)
        self.filter_convs,self.gate_convs,self.skip_convs,self.bn,self.residual_convs,self.gat_layers,self.gat_layers1=(nn.ModuleList() for _ in range(7))
        for index in range(blocks*layers):
            dilation=2**(index%layers);remaining-=(kernel_size-1)*dilation
            self.filter_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dilation)))
            self.gate_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dilation)))
            self.skip_convs.append(nn.Conv2d(dilation_channels,skip_channels,1))
            if index<blocks*layers-1:
                self.bn.append(SingletonBatchNorm2d(residual_channels))
                if run_gconv:
                    arguments=(dilation_channels*remaining,heads,feat_drop,attn_drop,negative_slope)
                    self.gat_layers.append(EdgeAttention(*arguments));self.gat_layers1.append(EdgeAttention(*arguments))
                else:
                    self.residual_convs.append(nn.Conv2d(dilation_channels,residual_channels,1))
        self.end_conv_1=nn.Conv2d(skip_channels,end_channels,1)
        self.end_conv_2=nn.Conv2d(end_channels,output_window*self.outputs,1)

    def forward(self,batch):
        source=batch['X']
        if source.ndim!=4 or source.shape[1:]!=(self.steps,self.nodes,self.features):
            raise ValueError('STMGAT X must match [B, history, nodes, features]')
        values=F.pad(source.permute(0,3,2,1),(max(0,self.receptive_field-self.steps),0))
        values=self.start_conv(values)+F.leaky_relu(self.cat_feature_conv(values))
        skip=None;last=len(self.filter_convs)-1
        for index in range(last+1):
            residual=values
            values=self.filter_convs[index](values).tanh()*self.gate_convs[index](values).sigmoid()
            current=self.skip_convs[index](values)[...,-1:]
            skip=current if skip is None else skip+current
            if index==last: break
            if self.run_gconv:
                graph_input=values.transpose(1,2).flatten(2)
                graph_output=self.gat_layers1[index](self.gat_layers[index](graph_input,self.edge_index),self.edge_index)
                values=values+graph_output.reshape(len(source),self.nodes,values.shape[1],values.shape[-1]).transpose(1,2)
            else:
                values=self.residual_convs[index](values)
            values=self.bn[index](values+residual[...,-values.shape[-1]:])
        prediction=self.end_conv_2(self.end_conv_1(skip.relu()).relu())
        return prediction.reshape(len(source),self.horizon,self.outputs,self.nodes).transpose(-1,-2)
