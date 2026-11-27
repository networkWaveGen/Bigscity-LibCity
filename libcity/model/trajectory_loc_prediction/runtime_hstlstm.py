"""Spatial-temporal LSTM with train-fitted slots and an effective memory cell."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class STLSTMCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        for name,shape in (('w_ih',(4*hidden_size,input_size)),('w_hh',(4*hidden_size,hidden_size)),
                ('w_s',(3*hidden_size,input_size)),('w_q',(3*hidden_size,input_size)),
                ('b_ih',(4*hidden_size,)),('b_hh',(4*hidden_size,))):
            parameter = nn.Parameter(torch.empty(shape))
            nn.init.uniform_(parameter,-1/math.sqrt(hidden_size),1/math.sqrt(hidden_size))
            self.register_parameter(name,parameter)

    def forward(self,location,temporal,spatial,state):
        hidden,memory = state
        incoming,forget,candidate,outgoing = (F.linear(location,self.w_ih,self.b_ih)
            +F.linear(hidden,self.w_hh,self.b_hh)).chunk(4,-1)
        intervals = F.linear(temporal,self.w_s)+F.linear(spatial,self.w_q)
        temporal_in,temporal_forget,temporal_out = intervals.chunk(3,-1)
        incoming = (incoming+temporal_in).sigmoid()
        forget = (forget+temporal_forget).sigmoid()
        outgoing = (outgoing+temporal_out).sigmoid()
        memory = forget*memory+incoming*candidate.tanh()
        return outgoing*memory.tanh(),memory


class STLSTM(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.cell = STLSTMCell(input_size,hidden_size)

    def forward(self,locations,temporal,spatial,valid):
        hidden = locations.new_zeros(len(locations),self.cell.hidden_size)
        memory = torch.zeros_like(hidden)
        for index in range(locations.shape[1]):
            next_hidden,next_memory = self.cell(locations[:,index],temporal[:,index],spatial[:,index],(hidden,memory))
            keep = valid[:,index,None]
            hidden,memory = torch.where(keep,next_hidden,hidden),torch.where(keep,next_memory,memory)
        return hidden


class HSTLSTM(nn.Module):
    def __init__(self,*,metadata,tim_slot_len=100,dis_slot_len=100,embed_size=256,hidden_size=512):
        super().__init__()
        if min(embed_size,hidden_size)<1 or min(tim_slot_len,dis_slot_len)<2:
            raise ValueError('HSTLSTM requires positive widths and at least two temporal/spatial slots')
        info = metadata.info
        tim_max,dis_max = info['tim_slot_max'],info['dis_slot_max']
        if not all(math.isfinite(value) and value>0 for value in (tim_max,dis_max)):
            raise ValueError('HSTLSTM slot bounds must be finite and positive')
        self.tim_slot_len,self.dis_slot_len = min(tim_slot_len,int(tim_max)+1),min(dis_slot_len,int(dis_max)+1)
        # Keep fractional slot boundaries instead of truncating them to integers.
        self.register_buffer('tim_slots',torch.linspace(0,tim_max,self.tim_slot_len))
        self.register_buffer('dis_slots',torch.linspace(0,dis_max,self.dis_slot_len))
        self.embed_s = nn.Embedding(self.tim_slot_len,embed_size)
        self.embed_q = nn.Embedding(self.dis_slot_len,embed_size)
        nn.init.normal_(self.embed_s.weight,0,.1)
        nn.init.normal_(self.embed_q.weight,0,.1)
        self.embed_l = nn.Embedding(info['loc_size'],embed_size,padding_idx=info['loc_pad'])
        self.st_lstm = STLSTM(embed_size,hidden_size)
        self.linear = nn.Linear(hidden_size,info['loc_size'])

    @staticmethod
    def interpolate(values,slots,embedding):
        values = values.to(embedding.weight.dtype).clamp(0,slots[-1])
        upper = torch.searchsorted(slots,values.contiguous(),right=True).clamp(1,len(slots)-1)
        lower = upper-1
        ratio = ((values-slots[lower])/(slots[upper]-slots[lower]))[...,None]
        return embedding(lower)*(1-ratio)+embedding(upper)*ratio

    def forward(self,batch):
        locations,lengths = batch['current_loc'],batch['lengths']
        temporal,spatial = batch['tim_interval'],batch['dis']
        if (locations.ndim!=2 or temporal.shape!=locations.shape or spatial.shape!=locations.shape
                or lengths.shape!=(len(locations),) or (lengths<1).any() or (lengths>locations.shape[1]).any()):
            raise ValueError('HSTLSTM expects matching padded locations/intervals and valid lengths')
        stop = int(lengths.max())
        locations,temporal,spatial = locations[:,:stop],temporal[:,:stop],spatial[:,:stop]
        valid = torch.arange(stop,device=locations.device)[None]<lengths[:,None]
        locations = locations.masked_fill(~valid,0)
        temporal,spatial = temporal.masked_fill(~valid,0.),spatial.masked_fill(~valid,0.)
        if ((locations[valid]<1).any() or (locations[valid]>=self.embed_l.num_embeddings).any()
                or not torch.isfinite(temporal).all() or not torch.isfinite(spatial).all()
                or (temporal<0).any() or (spatial<0).any()):
            raise ValueError('HSTLSTM requires valid locations and finite nonnegative observed intervals')
        temporal = self.interpolate(temporal,self.tim_slots,self.embed_s)
        spatial = self.interpolate(spatial,self.dis_slots,self.embed_q)
        hidden = self.st_lstm(self.embed_l(locations),temporal,spatial,valid)
        return F.log_softmax(self.linear(hidden)[:,1:],dim=-1)
