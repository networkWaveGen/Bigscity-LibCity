"""Time/space transition matrices score each candidate without target-derived inputs."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class STRNN(nn.Module):
    def __init__(self,*,metadata,hidden_size=500,time_upper=0.,distance_upper=0.,candidate_chunk_size=512):
        super().__init__()
        if min(hidden_size,candidate_chunk_size)<1 or min(time_upper,distance_upper)<0:
            raise ValueError('STRNN requires positive widths/chunks and nonnegative bound overrides')
        self.up_time = time_upper or metadata.info['time_upper']
        self.up_loc = distance_upper or metadata.info['distance_upper']
        if any(not math.isfinite(value) or value<=0 for value in (self.up_time,self.up_loc)):
            raise ValueError('STRNN time/distance interpolation bounds must be finite and positive')
        self.chunk_size = candidate_chunk_size
        self.register_buffer('coordinates',torch.tensor(metadata.info['coordinates'].copy(),dtype=torch.float32))
        self.h0 = nn.Parameter(torch.empty(hidden_size,1))
        self.weight_ih = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.weight_th_upper = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.weight_th_lower = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.weight_sh_upper = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.weight_sh_lower = nn.Parameter(torch.empty(hidden_size,hidden_size))
        self.location_weight = nn.Embedding(metadata.info['loc_size'],hidden_size)
        self.permanet_weight = nn.Embedding(metadata.info['uid_size'],hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter,-1/math.sqrt(hidden_size),1/math.sqrt(hidden_size))

    def scores(self,batch):
        locations,elapsed,lengths = batch['current_loc'],batch['elapsed_hours'],batch['lengths']
        if locations.shape!=elapsed.shape or locations.ndim!=2 or lengths.shape!=(len(locations),):
            raise ValueError('STRNN locations, elapsed hours and lengths must have matching shapes')
        if (lengths<1).any() or (lengths>locations.shape[1]).any():
            raise ValueError('STRNN lengths must fit the observed prefix')
        valid = torch.arange(locations.shape[1],device=locations.device)[None]<lengths[:,None]
        if ((locations[valid]<1)|(locations[valid]>len(self.coordinates))).any():
            raise ValueError('STRNN observed locations must be nonpadding IDs in the POI vocabulary')
        if not torch.isfinite(elapsed[valid]).all():
            raise ValueError('STRNN observed elapsed times must be finite')
        safe_locations = locations.masked_fill(~valid,0)
        elapsed = elapsed.masked_fill(~valid,0.)
        query = elapsed.gather(1,(lengths-1)[:,None])
        if 'query_offset_hours' in batch:
            offset = batch['query_offset_hours']
            if offset.shape!=(len(locations),) or not torch.isfinite(offset).all() or (offset<0).any():
                raise ValueError('STRNN query_offset_hours must be a finite nonnegative vector')
            query = query+offset[:,None]
        gaps = query-elapsed
        if (gaps[valid]<0).any():
            raise ValueError('STRNN query must not precede observed prefix points')
        ratio = (gaps/self.up_time).clamp(0,1)[...,None]
        embeddings = self.location_weight(safe_locations)
        temporal = (1-ratio)*F.linear(embeddings,self.weight_th_upper)+ratio*F.linear(embeddings,self.weight_th_lower)
        upper = F.linear(temporal,self.weight_sh_upper).masked_fill(~valid[...,None],0.)
        difference = (F.linear(temporal,self.weight_sh_lower)-upper).masked_fill(~valid[...,None],0.)
        # Aggregate repeated history locations before candidate scoring. Geometry
        # is shared across the batch, avoiding a B×V×L×H intermediate.
        unique = safe_locations[valid].unique(sorted=True)
        indices = torch.searchsorted(unique,safe_locations).masked_fill(~valid,0)
        grouped = difference.new_zeros(len(locations),len(unique),difference.shape[-1])
        grouped = grouped.scatter_add(1,indices[...,None].expand_as(difference),difference)
        constant = upper.sum(1)+(self.weight_ih@self.h0).T
        user = self.permanet_weight(batch['uid'])
        observed = self.coordinates[unique-1]
        predictions = []
        for start in range(0,len(self.coordinates),self.chunk_size):
            stop = min(start+self.chunk_size,len(self.coordinates))
            with torch.autocast(self.coordinates.device.type,enabled=False):
                # Centered static coordinates retain small degree differences in FP32.
                distances = torch.cdist(self.coordinates[start:stop],observed,
                    compute_mode='donot_use_mm_for_euclid_dist')/self.up_loc
            latent = constant[:,None]+torch.einsum('vk,bkh->bvh',distances.clamp(0,1),grouped)
            hidden = latent.sigmoid()+user[:,None]
            candidates = self.location_weight.weight[start+1:stop+1]
            predictions.append((hidden*candidates[None]).sum(-1))
        return torch.cat(predictions,dim=1)

    def forward(self,batch):
        return torch.log_softmax(self.scores(batch),dim=-1)
