"""Per-sample hierarchical graphs; pure inference and explicit auxiliary features."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_graph import normalized_supports
from libcity.model.runtime_norm import SingletonBatchNorm2d


def feature_gram(features):
    values=features.float() if features.dtype in (torch.float16,torch.bfloat16) else features
    with torch.autocast(features.device.type,enabled=False):
        vectors=values.transpose(1,2).flatten(2)
        return vectors@vectors.transpose(-1,-2)


def global_supports(features,count):
    adjacency=(feature_gram(features)-.5).relu().detach()
    result=[]
    for value in (adjacency,adjacency.transpose(-1,-2))[:count]:
        degree=value.sum(-1,keepdim=True)
        transition=value/torch.where(degree>0,degree,torch.ones_like(degree))
        result.append(transition.softmax(-1))
    return torch.stack(result)


class Linear(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__();self.mlp=nn.Conv2d(inputs,outputs,1)
    def forward(self,values): return self.mlp(values)


class GCN(nn.Module):
    def __init__(self,inputs,outputs,dropout,support_count,order):
        super().__init__();self.order,self.dropout=order,dropout
        self.mlp=Linear((support_count*order+1)*inputs,outputs)

    def forward(self,source,supports):
        values=[source]
        for support in supports:
            current=source
            for _ in range(self.order):
                current=(torch.einsum('bcvt,vw->bcwt',current,support) if support.ndim==2 else
                    torch.einsum('bcvt,bvw->bcwt',current,support))
                values.append(current)
        return F.dropout(self.mlp(torch.cat(values,dim=1)),self.dropout,self.training)


class CHGCN(nn.Module):
    def __init__(self,inputs,outputs,dropout,support_count,order,etas):
        super().__init__();self.gcn=GCN(inputs,outputs,dropout,support_count,order)
        self.etas=etas

    def forward(self,source,supports,regional_supports,membership,mapping):
        regional=torch.einsum('nr,bcnt->bcrt',membership,source)
        global_=torch.einsum('rg,bcrt->bcgt',mapping,regional)
        original=self.gcn(source,supports)
        regional=self.gcn(regional,regional_supports)
        global_=self.gcn(global_,global_supports(global_,len(supports)))
        n1,n2,n3,n4=self.etas
        regional=regional+n1*torch.einsum('rg,bcgt->bcrt',mapping,global_).relu()
        original=original+n2*torch.einsum('nr,bcrt->bcnt',membership,regional).relu()
        regional=regional+n3*torch.einsum('nr,bcnt->bcrt',membership,original).relu()
        global_=global_+n4*torch.einsum('rg,bcrt->bcgt',mapping,regional).relu()
        return original,regional,global_


class HIEST(nn.Module):
    def __init__(self,metadata,input_window,output_window,blocks=4,layers=2,apt_layer=True,kernel_size=2,
                 residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,global_nodes=15,
                 adjtype='doubletransition',order=2,dropout=.3,n1=1.,n2=1.,n3=1.,n4=1.):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,blocks=blocks,layers=layers,
                kernel_size=kernel_size,residual_channels=residual_channels,dilation_channels=dilation_channels,
                skip_channels=skip_channels,end_channels=end_channels,global_nodes=global_nodes).items():
            if type(value) is not int or value<1: raise ValueError('HIEST '+name+' must be a positive integer')
        if type(order) is not int or order<0 or type(apt_layer) is not bool or not 0<=dropout<1 or any(not math.isfinite(value) or value<0 for value in (n1,n2,n3,n4)):
            raise ValueError('HIEST order, adaptive layers, dropout or enhancement weights are invalid')
        if metadata.input_layout.get('kind')!='window': raise ValueError('HIEST requires continuous window metadata')
        mapping=np.array(metadata.static_data.get('region_membership'),dtype=np.float32,copy=True)
        if mapping.ndim!=2 or mapping.shape[0]!=metadata.num_nodes or not mapping.shape[1] or not np.isfinite(mapping).all() or np.any(mapping<0) or np.any(mapping.sum(1)<=0) or not np.allclose(mapping.sum(0),1.):
            raise ValueError('HIEST requires nonempty column-normalized overlapping region membership')
        self.steps,self.horizon,self.nodes,self.features,self.outputs=input_window,output_window,metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.layers=max(1,int(np.rint(np.log2(1+(input_window-1)/(blocks*(kernel_size-1)))))) if apt_layer and kernel_size>1 else layers
        self.blocks,self.global_nodes=blocks,global_nodes
        self.register_buffer('Mor_mx',torch.tensor(mapping))
        supports=normalized_supports(metadata.adjacency,adjtype)
        self.register_buffer('supports',supports)
        self.register_buffer('supports_r',(self.Mor_mx.T@supports@self.Mor_mx).softmax(-1))
        self.Mrg=nn.Parameter(torch.empty(mapping.shape[1],global_nodes));nn.init.kaiming_uniform_(self.Mrg)
        self.start_conv=nn.Conv2d(self.features,residual_channels,1)
        self.filter_convs,self.gate_convs,self.skip_convs,self.gconv,self.bn=(nn.ModuleList() for _ in range(5))
        count=blocks*self.layers;self.receptive_field=1+blocks*(kernel_size-1)*(2**self.layers-1)
        for index in range(count):
            dilation=2**(index%self.layers)
            self.filter_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dilation)))
            self.gate_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dilation)))
            self.skip_convs.append(nn.Conv2d(dilation_channels,skip_channels,1))
            self.gconv.append(CHGCN(dilation_channels,residual_channels,dropout,len(supports),order,(n1,n2,n3,n4)))
            if index<count-1: self.bn.append(SingletonBatchNorm2d(residual_channels))
        last=str(count-1)
        self.bnc=nn.ModuleDict({last:SingletonBatchNorm2d(residual_channels)})
        self.bns=nn.ModuleDict({last:SingletonBatchNorm2d(residual_channels)})
        self.end_conv_1=nn.Conv2d(skip_channels,end_channels,1)
        self.end_conv_2=nn.Conv2d(end_channels,output_window*self.outputs,1)

    def _forward(self,batch,auxiliary):
        source=batch['X']
        if source.ndim!=4 or source.shape[1:]!=(self.steps,self.nodes,self.features):
            raise ValueError('HIEST X must match [B, history, nodes, features]')
        source=F.pad(source.transpose(1,3),(1,0))
        source=F.pad(source,(max(0,self.receptive_field-source.shape[-1]),0))
        values=self.start_conv(source);mapping=self.Mrg.relu().softmax(1);skip=None;regional=global_=None
        last=len(self.gconv)-1
        for index,layer in enumerate(self.gconv):
            residual=values
            values=self.filter_convs[index](values).tanh()*self.gate_convs[index](values).sigmoid()
            current=self.skip_convs[index](values)[...,-1:]
            skip=current if skip is None else skip+current
            if index<last or auxiliary:
                values,regional,global_=layer(values,self.supports,self.supports_r,self.Mor_mx,mapping)
                if index<last: values=self.bn[index](values+residual[...,-values.shape[-1]:])
        prediction=self.end_conv_2(self.end_conv_1(skip.relu()).relu())
        prediction=prediction.reshape(len(source),self.horizon,self.outputs,self.nodes).transpose(-1,-2)
        if auxiliary:
            return prediction,self.bnc[str(last)](regional),self.bns[str(last)](global_),mapping
        return prediction

    def forward(self,batch): return self._forward(batch,False)
    def forward_with_aux(self,batch): return self._forward(batch,True)
