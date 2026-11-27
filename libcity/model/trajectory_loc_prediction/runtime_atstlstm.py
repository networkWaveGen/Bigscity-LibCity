"""Attentive personalized sequence scores without target-derived query features."""
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class Attention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.sqrt_rec_size = 1/math.sqrt(hidden_size)
        self.zu = nn.Linear(hidden_size,1,bias=False)

    def forward(self,values,valid):
        logits = (self.zu(values)*self.sqrt_rec_size).squeeze(-1).masked_fill(~valid,-torch.inf)
        return (logits.softmax(-1)[...,None]*values).sum(1)


class ATSTLSTM(nn.Module):
    def __init__(self,*,metadata,hidden_size=200):
        super().__init__()
        if hidden_size<1:
            raise ValueError('ATSTLSTM requires a positive hidden_size')
        info = metadata.info
        self.loc_embedding = nn.Embedding(info['loc_size'],hidden_size,padding_idx=info['loc_pad'])
        self.user_embedding = nn.Embedding(info['uid_size'],hidden_size)
        self.wv = nn.Linear(hidden_size,hidden_size,bias=False)
        self.wl = nn.Linear(1,hidden_size,bias=False)
        self.wt = nn.Linear(1,hidden_size,bias=False)
        self.wn = nn.Linear(hidden_size,hidden_size,bias=False)
        self.wp = nn.Linear(hidden_size,hidden_size,bias=False)
        self.lstm = nn.LSTM(hidden_size,hidden_size,batch_first=True)
        self.attn = Attention(hidden_size)

    def encode(self,batch):
        locations,lengths = batch['current_loc'],batch['lengths']
        temporal,spatial = batch['tim_interval'],batch['dis']
        distances = batch['candidate_dis']
        if (locations.ndim!=2 or temporal.shape!=locations.shape or spatial.shape!=locations.shape
                or lengths.shape!=(len(locations),) or (lengths<1).any() or (lengths>locations.shape[1]).any()
                or distances.shape!=(len(locations),self.loc_embedding.num_embeddings-1)):
            raise ValueError('ATSTLSTM expects matching padded prefixes, valid lengths and full candidate distances')
        stop = int(lengths.max())
        locations,temporal,spatial = locations[:,:stop],temporal[:,:stop],spatial[:,:stop]
        valid = torch.arange(stop,device=locations.device)[None]<lengths[:,None]
        locations = locations.masked_fill(~valid,0)
        temporal,spatial = temporal.masked_fill(~valid,0.),spatial.masked_fill(~valid,0.)
        offset = batch.get('query_offset_seconds',temporal.new_zeros(len(locations)))
        if ((locations[valid]<1).any() or (locations[valid]>=self.loc_embedding.num_embeddings).any()
                or offset.shape!=(len(locations),) or not torch.isfinite(offset).all() or (offset<0).any()):
            raise ValueError('ATSTLSTM requires valid locations and finite nonnegative query_offset_seconds')
        if any(not torch.isfinite(values).all() or (values<0).any() for values in (temporal,spatial,distances)):
            raise ValueError('ATSTLSTM requires finite nonnegative observed intervals and candidate distances')
        dtype = self.loc_embedding.weight.dtype
        temporal,spatial,distances,offset = (value.to(dtype) for value in (temporal,spatial,distances,offset))
        embedded = self.wv(self.loc_embedding(locations))+self.wl(spatial[...,None])+self.wt(temporal[...,None])
        packed = pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,_ = self.lstm(packed)
        encoded,_ = pad_packed_sequence(encoded,batch_first=True,total_length=stop)
        context = self.wn(self.attn(encoded,valid))+self.wp(self.user_embedding(batch['uid']))
        return context,distances,offset

    def score(self,context,distances,offset,candidates=None):
        # Factor the candidate projection: no B x vocabulary x hidden tensor.
        projected = context@self.wv.weight
        if candidates is None:
            scores = projected@self.loc_embedding.weight[1:].T
        else:
            if (candidates.ndim!=2 or candidates.shape[0]!=len(context) or (candidates<0).any()
                    or (candidates>=self.loc_embedding.num_embeddings-1).any()):
                raise ValueError('ATSTLSTM candidate indices must belong to the nonpadding vocabulary')
            distances = distances.gather(1,candidates)
            scores = (self.loc_embedding(candidates+1)*projected[:,None]).sum(-1)
        scores = scores+(context@self.wl.weight).squeeze(-1)[:,None]*distances
        return scores+(context@self.wt.weight).squeeze(-1)[:,None]*offset[:,None]

    def forward(self,batch):
        return self.score(*self.encode(batch))

    def forward_candidates(self,batch,candidates):
        # Training selects positives/negatives outside the inference interface.
        return self.score(*self.encode(batch),candidates)
