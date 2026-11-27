"""Context-aware recurrence reused once per prefix, then once per candidate."""
import torch
from torch import nn

from libcity.model.trajectory_loc_prediction.runtime_geography import spherical_distance


class CARACell(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.width = width
        for name in ('W_z','U_z','W_r','U_r','W_h','U_h','A_h','A_u','W_t','W_g'):
            value = nn.Parameter(torch.empty(width,width))
            nn.init.xavier_uniform_(value)
            self.register_parameter(name,value)
        for name in ('U_t','U_g'):
            value = nn.Parameter(torch.empty(1,width))
            nn.init.xavier_uniform_(value)
            self.register_parameter(name,value)
        for name in ('b_z','b_r','b_h','b_a_h','b_a_u','b_t','b_g'):
            self.register_parameter(name,nn.Parameter(torch.zeros(width)))

    @staticmethod
    def gate(values):
        # The author's Keras 1.2 hard sigmoid has slope 0.2, not PyTorch's 1/6.
        return (.2*values+.5).clamp(0,1)

    def forward(self,location,calendar,gap,distance,hidden):
        temporal = self.gate(gap@self.U_t)
        spatial = self.gate(distance@self.U_g)
        temporal_gate = self.gate(location@self.W_t+temporal+self.b_t)
        spatial_gate = self.gate(location@self.W_g+spatial+self.b_g)
        attention = self.gate(hidden@self.A_h+calendar@self.A_u+self.b_a_h+self.b_a_u)
        context = attention*calendar
        complement = (1-attention)*calendar
        update = self.gate(location@self.W_z+self.b_z+hidden@self.U_z+context@self.W_z+self.b_z)
        reset = self.gate(location@self.W_r+self.b_r+hidden@self.U_r+context@self.W_r+self.b_r)
        candidate = (location@self.W_h+self.b_h+
            (reset*temporal_gate*spatial_gate*hidden)@self.U_h+context@self.W_h+self.b_h).tanh()
        correction = 1+complement@self.W_z+self.b_z+complement@self.W_r+self.b_r+complement@self.W_h+self.b_h
        return correction*(update*hidden+(1-update)*candidate)


class CARA(nn.Module):
    def __init__(self,*,metadata,latent_dim=10,history_window=0,candidate_chunk_size=128):
        super().__init__()
        if min(latent_dim,candidate_chunk_size)<1 or history_window<0:
            raise ValueError('CARA requires positive widths/chunks and a nonnegative history_window')
        self.history_window,self.chunk_size = history_window,candidate_chunk_size
        info = metadata.info
        self.register_buffer('coordinates',torch.tensor(info['coordinates'].copy(),dtype=torch.float32))
        self.register_buffer('coordinate_origin',torch.tensor(info['coordinate_origin'].copy(),dtype=torch.float32))
        self.U_Embedding = nn.Embedding(info['uid_size'],latent_dim)
        self.V_Embedding = nn.Embedding(info['loc_size'],latent_dim,padding_idx=0)
        self.T_Embedding = nn.Embedding(info['tim_size'],latent_dim,padding_idx=info['tim_pad'])
        for module in (self.U_Embedding,self.V_Embedding,self.T_Embedding):
            nn.init.normal_(module.weight,0,.01)
        self.cell = CARACell(latent_dim)

    def distances(self,left,right):
        return spherical_distance(left,right,self.coordinate_origin,radius=6373)

    def encode(self,batch):
        locations,times,lengths,gaps,users,week = (batch[key] for key in
            ('current_loc','current_tim','lengths','time_gaps','uid','last_week_hour'))
        if (locations.ndim!=2 or times.shape!=locations.shape or gaps.shape!=locations.shape
                or lengths.shape!=(len(locations),) or users.shape!=(len(locations),) or week.shape!=(len(locations),)
                or (lengths<1).any() or (lengths>locations.shape[1]).any()):
            raise ValueError('CARA expects matching padded prefixes, valid lengths, users and known calendar times')
        valid = torch.arange(locations.shape[1],device=locations.device)[None]<lengths[:,None]
        locations,times = locations.masked_fill(~valid,0),times.masked_fill(~valid,48)
        gaps = gaps.masked_fill(~valid,0.)
        offset = batch.get('query_offset_hours',gaps.new_zeros(len(locations)))
        if ((locations[valid]<1).any() or (locations[valid]>=self.V_Embedding.num_embeddings).any()
                or (times[valid]<0).any() or (times[valid]>=48).any()
                or (users<0).any() or (users>=self.U_Embedding.num_embeddings).any()
                or not torch.isfinite(gaps).all() or (gaps<0).any() or not torch.isfinite(week).all()
                or (week<0).any() or (week>=168).any() or offset.shape!=(len(locations),)
                or not torch.isfinite(offset).all() or (offset<0).any()):
            raise ValueError('CARA requires valid observed tokens and finite nonnegative gaps/query_offset_hours')
        stop = min(int(lengths.max()),self.history_window) if self.history_window else int(lengths.max())
        steps = torch.arange(stop,device=locations.device)[None]
        start = (lengths-stop).clamp_min(0)
        indices = start[:,None]+steps
        locations,times,gaps = (values.gather(1,indices) for values in (locations,times,gaps))
        lengths = lengths.clamp_max(stop)
        valid = steps<lengths[:,None]
        # The first retained observation starts a fresh recurrent context.
        gaps = gaps.clone();gaps[:,0] = 0.
        gaps = gaps.to(self.coordinates.dtype)
        observed = self.coordinates[(locations-1).clamp_min(0)]
        previous = torch.cat((observed[:,:1],observed[:,:-1]),dim=1)
        distances = self.distances(observed,previous)
        location,calendar = self.V_Embedding(locations),self.T_Embedding(times)
        hidden = location.new_zeros(len(locations),self.cell.width)
        for index in range(stop):
            next_hidden = self.cell(location[:,index],calendar[:,index],gaps[:,index,None],distances[:,index,None],hidden)
            hidden = torch.where(valid[:,index,None],next_hidden,hidden)
        last = observed.gather(1,(lengths-1)[:,None,None].expand(-1,1,2)).squeeze(1)
        query_week = (week.to(self.coordinates.dtype)+offset.to(self.coordinates.dtype))%168
        query_calendar = (query_week%24).floor().long()+24*(query_week>=120).long()
        return hidden,last,self.T_Embedding(query_calendar),offset.to(hidden.dtype),self.U_Embedding(users)

    def score(self,hidden,last,calendar,gap,user,candidates):
        if candidates.ndim==1:
            candidates = candidates[None].expand(len(hidden),-1)
        if (candidates.ndim!=2 or candidates.shape[0]!=len(hidden) or not candidates.shape[1]
                or (candidates<0).any() or (candidates>=len(self.coordinates)).any()):
            raise ValueError('CARA candidates must belong to the nonpadding location vocabulary')
        predictions = []
        for start in range(0,candidates.shape[1],self.chunk_size):
            chosen = candidates[:,start:start+self.chunk_size]
            count = chosen.shape[1]
            distances = self.distances(last[:,None],self.coordinates[chosen])
            candidate_hidden = self.cell(self.V_Embedding(chosen+1).reshape(-1,self.cell.width),
                calendar[:,None].expand(-1,count,-1).reshape(-1,self.cell.width),
                gap[:,None,None].expand(-1,count,1).reshape(-1,1),distances.reshape(-1,1),
                hidden[:,None].expand(-1,count,-1).reshape(-1,self.cell.width))
            predictions.append((candidate_hidden.reshape(len(hidden),count,-1)*user[:,None]).sum(-1))
        return torch.cat(predictions,dim=1)

    def forward(self,batch):
        candidates = torch.arange(len(self.coordinates),device=self.coordinates.device)
        return self.score(*self.encode(batch),candidates)

    def forward_candidates(self,batch,candidates):
        return self.score(*self.encode(batch),candidates)
