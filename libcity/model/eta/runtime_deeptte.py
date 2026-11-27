"""DeepTTE's attribute, geo-convolution, recurrent and two estimation branches.

Input geometry is a known planned route; observed per-point times are targets only.
Normalization belongs to the data/loss adapter, and forward always returns Tensor.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from libcity.model.eta.DeepTTE import EntireEstimator, LocalEstimator


class DeepTTE(nn.Module):
    def __init__(self,*,metadata,uid_emb_size=16,weekid_emb_size=3,timid_emb_size=8,
                 kernel_size=3,num_filter=32,hidden_size=64,num_final_fcs=2,final_fc_size=64,
                 rnn_type='LSTM',pooling_method='attention'):
        super().__init__()
        if kernel_size<2 or rnn_type not in ('RNN','LSTM') or pooling_method not in ('mean','attention'):
            raise ValueError('DeepTTE needs kernel_size>=2, RNN/LSTM and mean/attention pooling')
        self.kernel_size,self.pooling_method=kernel_size,pooling_method
        self.uid_em=nn.Embedding(metadata.info['uid_size'],uid_emb_size)
        self.weekid_em=nn.Embedding(7,weekid_emb_size)
        self.timeid_em=nn.Embedding(1440,timid_emb_size)
        self.state_em=nn.Embedding(2,2)
        attributes=uid_emb_size+weekid_emb_size+timid_emb_size+1
        self.process_coords=nn.Linear(4,16)
        self.conv=nn.Conv1d(16,num_filter,kernel_size)
        self.rnn=getattr(nn,rnn_type)(num_filter+1+attributes,hidden_size,batch_first=True)
        self.attr2atten=nn.Linear(attributes,hidden_size)
        self.entire_estimate=EntireEstimator(hidden_size+attributes,num_final_fcs,final_fc_size)
        self.local_estimate=LocalEstimator(hidden_size)

    def encode(self,batch):
        lengths=batch['lengths']-self.kernel_size+1
        if (lengths<=0).any():
            raise ValueError('Every ETA route must be at least kernel_size points long')
        attributes=torch.cat((self.uid_em(batch['uid']),self.weekid_em(batch['weekid']),
                              self.timeid_em(batch['timeid']),batch['dist']),dim=-1)
        positions=torch.cat((batch['coords'],self.state_em(batch['state'])),dim=-1)
        convolution=F.elu(self.conv(torch.tanh(self.process_coords(positions)).transpose(1,2))).transpose(1,2)
        local_distance=batch['distance_gaps'].unfold(1,self.kernel_size-1,1).sum(-1).unsqueeze(-1)
        source=torch.cat((convolution,local_distance,attributes[:,None].expand(-1,convolution.shape[1],-1)),dim=-1)
        packed=pack_padded_sequence(source,lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed,_=self.rnn(packed)
        hidden,_=pad_packed_sequence(packed,batch_first=True,total_length=convolution.shape[1])
        valid=torch.arange(hidden.shape[1],device=hidden.device)[None]<lengths[:,None]
        if self.pooling_method=='mean':
            pooled=(hidden*valid[...,None]).sum(1)/lengths[:,None]
        else:
            logits=-(hidden*torch.tanh(self.attr2atten(attributes))[:,None]).sum(-1)
            attention=torch.softmax(logits.masked_fill(~valid,float('-inf')),dim=1)
            pooled=(hidden*attention[...,None]).sum(1)
        return attributes,pooled,hidden,valid

    def forward(self,batch):
        attributes,pooled,_,_=self.encode(batch)
        return self.entire_estimate(attributes,pooled)

    def forward_with_aux(self,batch):
        attributes,pooled,hidden,valid=self.encode(batch)
        return self.entire_estimate(attributes,pooled),self.local_estimate(hidden[valid]),valid
