"""Two-level spatial-temporal attention with masked keys and causal query time."""
import math

import torch
from torch import nn


class STAN(nn.Module):
    def __init__(self,*,metadata,embed_dim=50,max_len=100,candidate_chunk_size=256,time_upper=0.,distance_upper=0.):
        super().__init__()
        if min(embed_dim,max_len,candidate_chunk_size)<1 or min(time_upper,distance_upper)<0:
            raise ValueError('STAN requires positive widths/length/chunks and nonnegative bound overrides')
        info = metadata.info
        if max_len<info.get('max_prefix_length',1):
            raise ValueError(f'STAN model.params.max_len must cover the longest prefix: {info["max_prefix_length"]}')
        self.temporal_upper = time_upper or info['temporal_upper']
        self.spatial_upper = distance_upper or info['spatial_upper']
        if not all(math.isfinite(value) and value>0 for value in (self.temporal_upper,self.spatial_upper)):
            raise ValueError('STAN time and distance bounds must be finite and positive')
        self.max_len,self.chunk_size = max_len,candidate_chunk_size
        self.register_buffer('coordinates',torch.tensor(info['coordinates'].copy(),dtype=torch.float32))
        self.register_buffer('coordinate_origin',torch.tensor(info['coordinate_origin'].copy(),dtype=torch.float32))
        self.emb_t = nn.Embedding(info['tim_size'],embed_dim,padding_idx=0)
        self.emb_l = nn.Embedding(info['loc_size'],embed_dim,padding_idx=0)
        self.emb_u = nn.Embedding(info['uid_size'],embed_dim,padding_idx=0)
        for name in ('emb_su','emb_sl','emb_tu','emb_tl'):
            setattr(self,name,nn.Embedding(2,embed_dim,padding_idx=0))
        self.query = nn.Linear(embed_dim,embed_dim,bias=False)
        self.key = nn.Linear(embed_dim,embed_dim,bias=False)
        self.value = nn.Linear(embed_dim,embed_dim,bias=False)
        self.position = nn.Linear(max_len,1,bias=False)

    def distances(self,left,right):
        # Centered coordinates keep small POI separations when the module is FP32.
        with torch.autocast(self.coordinates.device.type,enabled=False):
            difference = torch.deg2rad(left-right)
            lat1 = torch.deg2rad(left[...,1]+self.coordinate_origin[1])
            lat2 = torch.deg2rad(right[...,1]+self.coordinate_origin[1])
            a = (difference[...,1]/2).sin().square()+lat1.cos()*lat2.cos()*(difference[...,0]/2).sin().square()
            a = a.clamp(0,1)
            return 2*6371*torch.atan2(a.sqrt(),(1-a).sqrt())

    def interval(self,spatial,temporal):
        # The original code sums the embedding axis immediately. Collapse that
        # linear interpolation first instead of allocating B x L x L/V x E.
        space = (spatial/self.spatial_upper).clamp(0,1)
        time = (temporal/self.temporal_upper).clamp(0,1)
        return ((1-space)*self.emb_sl.weight[1].sum()+space*self.emb_su.weight[1].sum()
            +(1-time)*self.emb_tl.weight[1].sum()+time*self.emb_tu.weight[1].sum())

    def encode(self,batch):
        locations,times,lengths,elapsed,users = (batch[key] for key in ('current_loc','current_tim','lengths','elapsed_hours','uid'))
        if (locations.ndim!=2 or times.shape!=locations.shape or elapsed.shape!=locations.shape
                or lengths.shape!=(len(locations),) or users.shape!=(len(locations),)
                or (lengths<1).any() or (lengths>locations.shape[1]).any() or (lengths>self.max_len).any()):
            raise ValueError('STAN expects matching prefixes and lengths within model.params.max_len')
        stop = int(lengths.max())
        locations,times,elapsed = locations[:,:stop],times[:,:stop],elapsed[:,:stop]
        valid = torch.arange(stop,device=locations.device)[None]<lengths[:,None]
        locations,times = locations.masked_fill(~valid,0),times.masked_fill(~valid,0)
        elapsed = elapsed.masked_fill(~valid,0.)
        offset = batch.get('query_offset_hours',elapsed.new_zeros(len(locations)))
        if ((locations[valid]<1).any() or (locations[valid]>=self.emb_l.num_embeddings).any()
                or (times[valid]<1).any() or (times[valid]>=self.emb_t.num_embeddings).any()
                or (users<1).any() or (users>=self.emb_u.num_embeddings).any()
                or not torch.isfinite(elapsed).all() or (elapsed<0).any()
                or offset.shape!=(len(locations),) or not torch.isfinite(offset).all() or (offset<0).any()):
            raise ValueError('STAN requires valid observed tokens and finite nonnegative times/query_offset_hours')
        elapsed = elapsed.to(self.coordinates.dtype)
        query_time = elapsed.gather(1,(lengths-1)[:,None])+offset.to(elapsed.dtype)[:,None]
        temporal = (elapsed[:,:,None]-elapsed[:,None,:]).abs()
        observed = self.coordinates[(locations-1).clamp_min(0)]
        spatial = self.distances(observed[:,:,None],observed[:,None])
        delta = self.interval(spatial,temporal)
        joint = self.emb_t(times)+self.emb_l(locations)+self.emb_u(users)[:,None]
        logits = self.query(joint)@self.key(joint).transpose(-1,-2)+delta
        attention = logits.masked_fill(~valid[:,None,:],-torch.inf).softmax(-1)
        attention = attention.masked_fill(~valid[:,:,None],0.)
        encoded = attention@self.value(joint)
        candidate_time = (query_time-elapsed).abs()
        return encoded,observed,candidate_time,valid

    def score(self,encoded,observed,temporal,valid,candidates):
        if candidates.ndim!=1 or not len(candidates) or (candidates<0).any() or (candidates>=len(self.coordinates)).any():
            raise ValueError('STAN candidates must be nonempty indices in the location vocabulary')
        predictions = []
        for start in range(0,len(candidates),self.chunk_size):
            chosen = candidates[start:start+self.chunk_size]
            spatial = self.distances(observed[:,None],self.coordinates[chosen][None,:,None])
            delta = self.interval(spatial,temporal[:,None]).masked_fill(~valid[:,None,:],0.)
            similarity = torch.einsum('ce,ble->bcl',self.emb_l(chosen+1),encoded)
            weights = self.position.weight[0,:encoded.shape[1]]
            predictions.append((similarity*delta*weights).sum(-1))
        return torch.cat(predictions,dim=-1)

    def forward(self,batch):
        candidates = torch.arange(len(self.coordinates),device=self.coordinates.device)
        return torch.log_softmax(self.score(*self.encode(batch),candidates),dim=-1)

    def forward_candidates(self,batch,candidates):
        return self.score(*self.encode(batch),candidates)
