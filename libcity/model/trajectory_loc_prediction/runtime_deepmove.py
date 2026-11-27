"""Historical recurrent attention with explicit lengths and a masked cold start."""
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class Attention(nn.Module):
    def __init__(self,method,hidden):
        super().__init__()
        self.method = method
        if method=='general':
            self.attn = nn.Linear(hidden,hidden)
        elif method=='concat':
            self.attn = nn.Linear(2*hidden,hidden)
            self.other = nn.Parameter(torch.empty(hidden))
            nn.init.uniform_(self.other,-1/math.sqrt(hidden),1/math.sqrt(hidden))

    def forward(self,query,history,valid):
        if self.method=='concat':
            combined = torch.cat((query[:,None].expand_as(history),history),dim=-1)
            scores = (self.attn(combined).tanh()*self.other).sum(-1)
        else:
            values = self.attn(history) if self.method=='general' else history
            scores = (query[:,None]*values).sum(-1)
        scores = scores.masked_fill(~valid,-torch.inf)
        scores = torch.where(valid.any(dim=1,keepdim=True),scores,0.)
        return scores.softmax(dim=1).masked_fill(~valid,0.)


class DeepMove(nn.Module):
    def __init__(self,*,metadata,loc_emb_size=500,tim_emb_size=10,hidden_size=500,
                 attn_type='dot',rnn_type='LSTM',dropout_p=.5):
        super().__init__()
        if min(loc_emb_size,tim_emb_size,hidden_size)<1 or attn_type not in ('dot','general','concat'):
            raise ValueError('DeepMove requires positive widths and dot/general/concat attention')
        if rnn_type not in ('RNN','GRU','LSTM') or not 0<=dropout_p<1:
            raise ValueError('DeepMove requires RNN/GRU/LSTM and dropout in [0,1)')
        info = metadata.info
        self.loc_pad,self.tim_pad = info['loc_pad'],info['tim_pad']
        self.emb_loc = nn.Embedding(info['loc_size'],loc_emb_size,padding_idx=self.loc_pad)
        self.emb_tim = nn.Embedding(info['tim_size'],tim_emb_size,padding_idx=self.tim_pad)
        recurrent = getattr(nn,rnn_type)
        self.rnn_encoder = recurrent(loc_emb_size+tim_emb_size,hidden_size,batch_first=True)
        self.rnn_decoder = recurrent(loc_emb_size+tim_emb_size,hidden_size,batch_first=True)
        self.attn = Attention(attn_type,hidden_size)
        self.fc_final = nn.Linear(2*hidden_size,info['loc_size'])
        self.dropout = nn.Dropout(dropout_p)
        for name,parameter in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(parameter)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(parameter)
            elif 'bias' in name:
                nn.init.zeros_(parameter)

    def packed_inputs(self,locations,times,lengths,*,allow_empty=False):
        if locations.shape!=times.shape or locations.ndim!=2 or lengths.shape!=(len(locations),):
            raise ValueError('DeepMove token/time tensors and lengths must have matching batch shapes')
        if (lengths<(0 if allow_empty else 1)).any() or (lengths>locations.shape[1]).any():
            raise ValueError('DeepMove lengths must fit each padded sequence')
        if locations.shape[1]==0:
            locations = locations.new_full((len(locations),1),self.loc_pad)
            times = times.new_full((len(times),1),self.tim_pad)
        valid = torch.arange(locations.shape[1],device=locations.device)[None]<lengths[:,None]
        locations = locations.masked_fill(~valid,self.loc_pad)
        times = times.masked_fill(~valid,self.tim_pad)
        values = self.dropout(torch.cat((self.emb_loc(locations),self.emb_tim(times)),dim=-1))
        packed = pack_padded_sequence(values,lengths.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
        return packed

    def forward(self,batch):
        current = self.packed_inputs(batch['current_loc'],batch['current_tim'],batch['lengths'])
        history = self.packed_inputs(batch['history_loc'],batch['history_tim'],batch['history_lengths'],allow_empty=True)
        history,_ = self.rnn_encoder(history)
        history,_ = pad_packed_sequence(history,batch_first=True)
        _,state = self.rnn_decoder(current)
        final = (state[0] if isinstance(state,tuple) else state)[-1]
        valid = torch.arange(history.shape[1],device=history.device)[None]<batch['history_lengths'][:,None]
        weights = self.attn(final,history,valid)
        context = (weights[...,None]*history).sum(1)
        logits = self.fc_final(self.dropout(torch.cat((final,context),dim=-1)))
        return torch.log_softmax(logits[:,1:],dim=-1)
