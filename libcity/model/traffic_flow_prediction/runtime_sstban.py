"""Bottleneck attention with explicit heads, masks and training-only auxiliaries."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_norm import SingletonBatchNorm2d


class Pointwise(nn.Module):
    def __init__(self,inputs,outputs,relu,momentum):
        super().__init__();self.relu=relu
        self.conv=nn.Conv2d(inputs,outputs,1);self.batch_norm=SingletonBatchNorm2d(outputs,momentum=momentum)
        nn.init.xavier_uniform_(self.conv.weight);nn.init.zeros_(self.conv.bias)
    def forward(self,source):
        values=self.batch_norm(self.conv(source.permute(0,3,2,1)))
        return (values.relu() if self.relu else values).permute(0,3,2,1)


class FC(nn.Module):
    def __init__(self,widths,activations,momentum):
        super().__init__()
        self.convs=nn.ModuleList(Pointwise(left,right,relu,momentum) for left,right,relu in zip(widths,widths[1:],activations))
    def forward(self,source):
        for layer in self.convs: source=layer(source)
        return source


class STEmbedding(nn.Module):
    def __init__(self,slots,hidden,momentum):
        super().__init__();self.slots=slots
        self.FC_te=FC([7+slots,hidden,hidden],[True,False],momentum)
    def forward(self,spatial,tokens):
        categorical=torch.cat((F.one_hot(tokens[...,0],7),F.one_hot(tokens[...,1],self.slots)),-1)
        return spatial[None,None]+self.FC_te(categorical.to(spatial.dtype)[:,:,None])


def split_heads(source,heads,width,temporal):
    values=source.unflatten(-1,(heads,width))
    return values.permute(0,3,2,1,4) if temporal else values.permute(0,3,1,2,4)


def merge_heads(source,temporal):
    values=source.permute(0,3,2,1,4) if temporal else source.permute(0,2,3,1,4)
    return values.flatten(-2)


class MAB(nn.Module):
    def __init__(self,heads,width,inputs,outputs,momentum,temporal=False,mask_role='key'):
        super().__init__();self.heads,self.width=heads,width;self.temporal,self.mask_role=temporal,mask_role
        hidden=heads*width
        self.FC_q=FC([inputs,hidden],[True],momentum)
        self.FC_k=FC([inputs,hidden],[True],momentum)
        self.FC_v=FC([inputs,hidden],[True],momentum)
        self.FC=FC([hidden,outputs],[True],momentum)
    def forward(self,query,key,mask=None):
        q=split_heads(self.FC_q(query),self.heads,self.width,self.temporal)
        k=split_heads(self.FC_k(key),self.heads,self.width,self.temporal)
        v=split_heads(self.FC_v(key),self.heads,self.width,self.temporal)
        allowed=None
        if mask is not None:
            kept=split_heads(mask,self.heads,self.width,self.temporal).any(-1)
            allowed=kept.unsqueeze(-2 if self.mask_role=='key' else -1)
        # Boolean masks give zero attention for an empty neighborhood, including
        # fully masked queries. Keep batch and head axes separate throughout.
        result=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.)
        return self.FC(merge_heads(result,self.temporal))


class BottleneckAttention(nn.Module):
    def __init__(self,heads,width,history,nodes,references,momentum,temporal=False):
        super().__init__();hidden=heads*width
        shape=(1,references,nodes,2*hidden) if temporal else (1,history,references,2*hidden)
        self.I=nn.Parameter(torch.empty(shape));nn.init.xavier_uniform_(self.I)
        self.mab0=MAB(heads,width,2*hidden,2*hidden,momentum,temporal,'key')
        self.mab1=MAB(heads,width,2*hidden,hidden,momentum,temporal,'query')
    def forward(self,source,embedding,mask):
        values=torch.cat((source,embedding),-1)
        inducing=self.I.expand(len(source),-1,-1,-1)
        hidden=self.mab0(inducing,values,mask)
        return self.mab1(values,hidden,mask)


class STAttBlock(nn.Module):
    def __init__(self,heads,width,history,nodes,references,momentum,auxiliary=False):
        super().__init__();self.auxiliary=auxiliary
        self.spatialAttention=BottleneckAttention(heads,width,history,nodes,references,momentum)
        self.temporalAttention=BottleneckAttention(heads,width,history,nodes,references,momentum,True)
        if auxiliary:
            self.mask_token=nn.Parameter(torch.empty(1,1,1,heads*width));nn.init.xavier_uniform_(self.mask_token)
    def forward(self,source,embedding,mask=None):
        if self.auxiliary and mask is not None: source=torch.where(mask,source,self.mask_token)
        return source+(self.spatialAttention(source,embedding,mask)+self.temporalAttention(source,embedding,mask))


class TransformAttention(nn.Module):
    def __init__(self,heads,width,momentum):
        super().__init__();self.heads,self.width=heads,width;hidden=heads*width
        self.FC_q=FC([hidden,hidden],[True],momentum)
        self.FC_k=FC([hidden,hidden],[True],momentum)
        self.FC_v=FC([hidden,hidden],[True],momentum)
        self.FC=FC([hidden,hidden],[True],momentum)
    def forward(self,source,history,future):
        q=split_heads(self.FC_q(future),self.heads,self.width,True)
        k=split_heads(self.FC_k(history),self.heads,self.width,True)
        v=split_heads(self.FC_v(source),self.heads,self.width,True)
        return self.FC(merge_heads(F.scaled_dot_product_attention(q,k,v,dropout_p=0.),True))


class SSTBAN(nn.Module):
    def __init__(self,metadata,input_window,output_window,L=3,K=16,d=8,reference=3,bn_decay=.1):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,L=L,K=K,d=d,reference=reference).items():
            if type(value) is not int or value<1: raise ValueError('SSTBAN '+name+' must be a positive integer')
        if not math.isfinite(bn_decay) or not 0<bn_decay<=1: raise ValueError('SSTBAN bn_decay must be in (0,1]')
        history=metadata.input_layout.get('history_features',{});future=metadata.input_layout.get('future_features',{})
        if history.get('kind')!='calendar_positions' or history!=future or tuple(history.get('columns',()))!=('weekday','time_slot'):
            raise ValueError('SSTBAN requires explicit historical and future weekday/time-slot metadata')
        self.steps,self.horizon,self.nodes,self.features,self.outputs=input_window,output_window,metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.slots=history['steps_per_day'];self.hidden_dim=K*d
        self.SE=nn.Parameter(torch.empty(self.nodes,self.hidden_dim))
        self.STEmbedding=STEmbedding(self.slots,self.hidden_dim,bn_decay)
        arguments=(K,d,input_window,self.nodes,reference,bn_decay)
        self.STAttBlock_1=nn.ModuleList(STAttBlock(*arguments) for _ in range(L))
        self.STAttBlock_self=nn.ModuleList([STAttBlock(*arguments,auxiliary=True)])
        self.STAttBlock_2=nn.ModuleList(STAttBlock(K,d,output_window,self.nodes,reference,bn_decay) for _ in range(L))
        self.transformAttention=TransformAttention(K,d,bn_decay)
        self.FC_1=FC([self.features,self.hidden_dim,self.hidden_dim],[True,False],bn_decay)
        self.FC_2=FC([self.hidden_dim,self.hidden_dim,self.outputs],[True,False],bn_decay)
        for parameter in self.parameters():
            if parameter.ndim>1: nn.init.xavier_uniform_(parameter)

    def _embed(self,batch):
        source=batch['X']
        if source.ndim!=4 or source.shape[1:]!=(self.steps,self.nodes,self.features):
            raise ValueError('SSTBAN X must match [B, history, nodes, features]')
        history,future=batch.get('history_features'),batch.get('future_features')
        for name,tokens,length in (('history_features',history,self.steps),('future_features',future,self.horizon)):
            if tokens is None or tokens.shape!=(len(source),length,2) or tokens.dtype not in (torch.int32,torch.int64):
                raise ValueError('SSTBAN '+name+' must be integer [B, time, 2] weekday/slot tokens')
            if (tokens<0).any() or (tokens[...,0]>=7).any() or (tokens[...,1]>=self.slots).any():
                raise ValueError('SSTBAN '+name+' contains out-of-range calendar tokens')
        projected=self.FC_1(source);embedding=self.STEmbedding(self.SE,torch.cat((history,future),1))
        return projected,embedding[:,:self.steps],embedding[:,self.steps:]

    def _encode(self,source,embedding,mask=None):
        for layer in self.STAttBlock_1: source=layer(source,embedding,mask)
        return source

    def _decode(self,encoded,history,future):
        values=self.transformAttention(encoded,history,future)
        for layer in self.STAttBlock_2: values=layer(values,future)
        return self.FC_2(values)

    def forward(self,batch):
        source,history,future=self._embed(batch)
        return self._decode(self._encode(source,history),history,future)

    def forward_with_aux(self,batch,keep_mask):
        source,history,future=self._embed(batch)
        if keep_mask.shape!=source.shape or keep_mask.dtype!=torch.bool:
            raise ValueError('SSTBAN keep_mask must be boolean [B, history, nodes, hidden_dim]')
        complete=self._encode(source,history)
        prediction=self._decode(complete,history,future)
        corrupted=self._encode(source,history,keep_mask)
        for layer in self.STAttBlock_self: corrupted=layer(corrupted,history,keep_mask)
        return prediction,complete,corrupted
