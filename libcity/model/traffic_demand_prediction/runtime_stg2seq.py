"""Graph sequence forecasting keeps batch, time, node and channel axes explicit."""
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.traffic_demand_prediction.STG2Seq import Align
from libcity.model.traffic_flow_prediction.runtime_multiscale import spectral_basis


class GraphTemporalConvolution(nn.Module):
    def __init__(self,inputs,outputs,kernel,order):
        super().__init__()
        self.kernel,self.outputs = kernel,outputs
        self.align = Align(inputs,outputs)
        self.weights = nn.Parameter(torch.empty(2*outputs,inputs*kernel*order))
        self.biases = nn.Parameter(torch.zeros(2*outputs))
        nn.init.xavier_uniform_(self.weights)

    def forward(self,source,basis):
        # B,C,T,N -> B,T,N,C,Kt,Ks, preserving each local time/node context.
        propagated = torch.einsum('bctn,knm->bctmk',source,basis)
        padded = F.pad(propagated,(0,0,0,0,0,self.kernel-1))
        windows = padded.unfold(2,self.kernel,1).permute(0,2,3,1,5,4)
        convolved = F.linear(windows.flatten(-3),self.weights,self.biases).permute(0,3,1,2)
        value,gate = convolved.split(self.outputs,dim=1)
        return (value+self.align(source))*gate.sigmoid()


class GraphStack(nn.Module):
    def __init__(self,inputs,hidden,kernels,order):
        super().__init__()
        self.layers = nn.ModuleList(nn.ModuleList((GraphTemporalConvolution(inputs if index==0 else hidden,
            hidden,kernel,order),nn.BatchNorm2d(hidden))) for index,kernel in enumerate(kernels))

    def forward(self,source,basis):
        for convolution,normalization in self.layers:
            source = normalization(convolution(source,basis))
        return source


class TemporalAttention(nn.Module):
    def __init__(self,steps,nodes,hidden,external):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(steps,nodes*hidden,1))
        self.weight2 = nn.Parameter(torch.empty(external,steps))
        self.bias = nn.Parameter(torch.zeros(steps))
        nn.init.xavier_uniform_(self.weight1)
        if external:
            nn.init.xavier_uniform_(self.weight2)

    def forward(self,query,source):
        values = source.permute(0,2,3,1)
        score = torch.einsum('btf,tf->bt',values.flatten(2),self.weight1[...,0])
        attention = (score+query@self.weight2+self.bias).tanh().softmax(dim=1)
        return torch.einsum('bt,btnc->bcn',attention,values).unsqueeze(2)


class ChannelAttention(nn.Module):
    def __init__(self,nodes,hidden,external):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(hidden,nodes,1))
        self.weight2 = nn.Parameter(torch.empty(external,hidden))
        self.bias = nn.Parameter(torch.zeros(hidden))
        nn.init.xavier_uniform_(self.weight1)
        if external:
            nn.init.xavier_uniform_(self.weight2)

    def forward(self,query,source):
        values = source[:,:,0].transpose(1,2)
        score = torch.einsum('bnc,cn->bc',values,self.weight1[...,0])
        attention = (score+query@self.weight2+self.bias).tanh().softmax(dim=1)
        return torch.einsum('bc,bnc->bn',attention,values)[:,None,None,:]


class STG2Seq(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,window=3,dim_out=32,ks=3):
        super().__init__()
        if min(window,dim_out,ks)<1 or window>input_window or metadata.input_layout.get('kind')!='window':
            raise ValueError('STG2Seq requires positive widths/order and contiguous history with window <= input_window')
        self.window,self.horizon,self.outputs = window,output_window,metadata.output_dim
        self.future_indices = tuple(metadata.input_layout.get('future_feature_indices',()))
        external = len(self.future_indices)
        self.register_buffer('basis',spectral_basis(metadata.adjacency,ks))
        self.long_term_layer = GraphStack(self.outputs,dim_out,[3,3,3,3,3,2],ks)
        self.short_term_gcn = GraphStack(self.outputs,dim_out,[3,3,3],ks)
        self.attention_t = TemporalAttention(input_window+window,metadata.num_nodes,dim_out,external)
        self.attention_c = nn.ModuleList(ChannelAttention(metadata.num_nodes,dim_out,external) for _ in range(self.outputs))

    def _decode(self,batch,targets=None):
        source = batch['X'][...,:self.outputs].permute(0,3,1,2)
        if self.future_indices and 'future_features' not in batch:
            raise ValueError('STG2Seq requires explicit future_features for declared known covariates')
        long = self.long_term_layer(source,self.basis)
        history = source[:,:,-self.window:]
        predictions = []
        for position in range(self.horizon):
            query = batch['future_features'][:,position,0] if self.future_indices else source.new_zeros(len(source),0)
            short = self.short_term_gcn(history,self.basis)
            representation = self.attention_t(query,torch.cat((short,long),dim=2))
            prediction = torch.cat([head(query,representation) for head in self.attention_c],dim=1)
            predictions.append(prediction)
            following = prediction if targets is None else targets[:,position,:,:self.outputs].transpose(1,2).unsqueeze(2)
            history = torch.cat((history[:,:,1:],following),dim=2)
        return torch.cat(predictions,dim=2).permute(0,2,3,1)

    def forward(self,batch):
        return self._decode(batch)

    def forward_with_teacher(self,batch,targets):
        return self._decode(batch,targets)
