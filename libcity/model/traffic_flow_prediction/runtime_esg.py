"""Evolving dense graphs with multiscale temporal extractors and owned priors."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_norm import SingletonBatchNorm1d


class NodeFeaExtractor(nn.Module):
    def __init__(self,hidden,rows):
        super().__init__()
        self.conv1,self.conv2=nn.Conv1d(1,8,10),nn.Conv1d(8,16,10)
        self.bn1,self.bn2,self.bn3=SingletonBatchNorm1d(8),SingletonBatchNorm1d(16),SingletonBatchNorm1d(hidden)
        self.fc=nn.Linear((rows-18)*16,hidden)

    def forward(self,history):
        values=history.T.unsqueeze(1)
        values=self.bn1(self.conv1(values).relu())
        values=self.bn2(self.conv2(values).relu())
        return self.bn3(self.fc(values.flatten(1)).relu())


class PairProjection(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.fc1,self.fc2=nn.Linear(hidden,1),nn.Linear(2*hidden,hidden)

    def forward(self,states):
        left,right=self.fc2.weight.split(states.shape[-1],dim=-1)
        # Algebraically identical to Linear(cat(receiver, sender)), without
        # repeating or materializing a B*N*N*2D concatenated pair tensor.
        pair=F.linear(states,left,self.fc2.bias)[:,:,None]+F.linear(states,right)[:,None,:]
        return self.fc1(pair.relu()).squeeze(-1)


class EvolvingGraphLearner(nn.Module):
    def __init__(self,features,hidden):
        super().__init__()
        self.rz_gate,self.h_candidate=nn.Linear(features+hidden,2*hidden),nn.Linear(features+hidden,hidden)
        self.conv,self.conv2=PairProjection(hidden),PairProjection(hidden)

    def forward(self,inputs,states):
        reset,update=self.rz_gate(torch.cat((inputs,states),dim=-1)).sigmoid().chunk(2,dim=-1)
        candidate=self.h_candidate(torch.cat((inputs,reset*states),dim=-1)).tanh()
        state=update*states+(1-update)*candidate
        values=state.relu()
        return self.conv(values)*self.conv2(values).sigmoid(),state


class Inception(nn.Module):
    def __init__(self,inputs,outputs,kernels,dilation):
        super().__init__()
        self.tconv=nn.ModuleList([nn.Conv2d(inputs,outputs//len(kernels),(1,kernel),dilation=(1,dilation)) for kernel in kernels])

    def forward(self,source):
        values=[layer(source) for layer in self.tconv];length=values[-1].shape[-1]
        return torch.cat([value[...,-length:] for value in values],dim=1)


class TConv(nn.Module):
    def __init__(self,residual,channels,kernels,dilation,dropout):
        super().__init__()
        self.filter_conv,self.gate_conv=Inception(residual,channels,kernels,dilation),Inception(residual,channels,kernels,dilation)
        self.dropout=dropout

    def forward(self,source):
        values=self.filter_conv(source).tanh()*self.gate_conv(source).sigmoid()
        return F.dropout(values,self.dropout,self.training)


class Projection(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__();self.mlp=nn.Conv2d(inputs,outputs,1)
    def forward(self,values): return self.mlp(values)


class MixProp(nn.Module):
    def __init__(self,inputs,outputs,depth,alpha):
        super().__init__();self.depth,self.alpha=depth,alpha
        self.mlp=Projection((depth+1)*inputs,outputs)

    def forward(self,source,graph):
        values=source;layers=[source]
        for _ in range(self.depth):
            values=self.alpha*source+(1-self.alpha)*torch.einsum('bcwt,bvw->bcvt',values,graph)
            layers.append(values)
        return self.mlp(torch.cat(layers,dim=1))


class EvolvingGConv(nn.Module):
    def __init__(self,channels,residual,depth,static,dynamic,interval,alpha):
        super().__init__();self.interval,self.depth=interval,depth
        self.linear_s2d=nn.Linear(static,dynamic) if depth else None
        self.scale_spc_EGL=EvolvingGraphLearner(channels,dynamic) if depth else None
        self.gconv=MixProp(channels,residual,depth,alpha)

    def forward(self,source,features):
        if not self.depth: return self.gconv(source,None)
        states=self.linear_s2d(features)[None].expand(len(source),-1,-1)
        outputs=[]
        for start in range(0,source.shape[-1],self.interval):
            values=source[...,start:start+self.interval]
            graph,states=self.scale_spc_EGL(values.mean(-1).transpose(1,2),states)
            outputs.append(self.gconv(values,graph))
        return torch.cat(outputs,dim=-1)


class Extractor(nn.Module):
    def __init__(self,residual,channels,kernels,dilation,depth,static,dynamic,skip,length,nodes,affine,alpha,dropout,interval):
        super().__init__()
        self.t_conv=TConv(residual,channels,kernels,dilation,dropout)
        self.skip_conv=nn.Conv2d(channels,skip,(1,length))
        self.s_conv=EvolvingGConv(channels,residual,depth,static,dynamic,interval,alpha)
        self.norm=nn.LayerNorm((residual,nodes,length),elementwise_affine=affine)

    def forward(self,source,features):
        values=self.t_conv(source)
        skip=self.skip_conv(values)
        output=self.s_conv(values,features)
        return self.norm(output+source[...,-output.shape[-1]:]),skip


class ESG(nn.Module):
    def __init__(self,metadata,input_window,output_window,n_blocks=1,n_layers=3,dropout=.3,residual_channels=32,
                 end_channels=128,st_embedding_dim=40,kernel_set=(2,6),dilation_exp=1,conv_channels=32,gcn_depth=2,
                 dy_embedding_dim=20,skip_channels=64,layer_norm_affline=False,propalpha=.05,dy_interval=(1,1,1)):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,n_blocks=n_blocks,n_layers=n_layers,
                residual_channels=residual_channels,end_channels=end_channels,st_embedding_dim=st_embedding_dim,
                dilation_exp=dilation_exp,conv_channels=conv_channels,dy_embedding_dim=dy_embedding_dim,skip_channels=skip_channels).items():
            if type(value) is not int or value<1: raise ValueError('ESG '+name+' must be a positive integer')
        if (not isinstance(kernel_set,(tuple,list)) or not kernel_set or any(type(v) is not int or v<1 for v in kernel_set) or
                any(b<=a for a,b in zip(kernel_set,kernel_set[1:])) or conv_channels%len(kernel_set)):
            raise ValueError('ESG requires increasing positive kernels and channels divisible by their number')
        if not isinstance(dy_interval,(tuple,list)) or len(dy_interval)!=n_layers or any(type(v) is not int or v<1 for v in dy_interval):
            raise ValueError('ESG dy_interval must contain one positive interval per layer')
        if type(gcn_depth) is not int or gcn_depth<0 or not 0<=dropout<1 or not 0<=propalpha<=1 or type(layer_norm_affline) is not bool:
            raise ValueError('ESG graph depth, dropout, propagation alpha or affine switch is invalid')
        if metadata.input_layout.get('kind')!='window': raise ValueError('ESG requires continuous window metadata')
        self.steps,self.horizon,self.nodes,self.features,self.outputs=input_window,output_window,metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.dropout=dropout
        self.receptive_field=1+n_blocks*(kernel_set[-1]-1)*sum(dilation_exp**layer for layer in range(n_layers))
        self.total=max(input_window,self.receptive_field)
        if gcn_depth:
            history=metadata.static_data.get('graph_history')
            if history is None or history.ndim!=2 or history.shape[1]!=self.nodes or history.shape[0]<19 or not np.isfinite(history).all():
                raise ValueError('ESG graph history requires at least 19 finite interleaved training rows and aligned nodes')
            self.register_buffer('static_feat',torch.tensor(np.array(history,copy=True),dtype=torch.float32))
            self.stfea_encode=NodeFeaExtractor(st_embedding_dim,len(history))
        else:
            self.register_buffer('static_feat',None);self.stfea_encode=None
        self.start_conv=nn.Conv2d(self.features,residual_channels,1)
        self.blocks=nn.ModuleList();length=self.total
        for _ in range(n_blocks):
            block=nn.ModuleList()
            for index in range(n_layers):
                dilation=dilation_exp**index;length-=(kernel_set[-1]-1)*dilation
                block.append(Extractor(residual_channels,conv_channels,kernel_set,dilation,gcn_depth,st_embedding_dim,
                    dy_embedding_dim,skip_channels,length,self.nodes,layer_norm_affline,propalpha,dropout,dy_interval[index]))
            self.blocks.append(block)
        self.skip0=nn.Conv2d(self.features,skip_channels,(1,self.total))
        self.skipE=nn.Conv2d(residual_channels,skip_channels,(1,length))
        self.out=nn.Sequential(nn.ReLU(),nn.Conv2d(skip_channels,end_channels,1),nn.ReLU(),nn.Conv2d(end_channels,output_window*self.outputs,1))

    def forward(self,batch):
        source=batch['X']
        if source.ndim!=4 or source.shape[1:]!=(self.steps,self.nodes,self.features):
            raise ValueError('ESG X must match [B, history, nodes, features]')
        source=source.permute(0,3,2,1)
        if self.total>self.steps: source=F.pad(source,(self.total-self.steps,0,0,0),mode='replicate')
        values=self.start_conv(source)
        features=self.stfea_encode(self.static_feat) if self.stfea_encode is not None else None
        skip=self.skip0(F.dropout(source,self.dropout,self.training))
        for block in self.blocks:
            for layer in block:
                values,current=layer(values,features);skip=skip+current
        prediction=self.out(skip+self.skipE(values))
        return prediction.reshape(len(source),self.horizon,self.outputs,self.nodes).transpose(-1,-2)
