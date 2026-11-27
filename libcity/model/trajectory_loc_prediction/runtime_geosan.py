"""Geographic self-attention with explicit prefix and candidate interfaces."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Embedding(nn.Module):
    def __init__(self,count,width,*,padding=True,scale=True):
        super().__init__()
        self.padding,self.scale,self.width = padding,scale,width
        self.lookup_table = nn.Parameter(torch.empty(count,width))
        nn.init.xavier_normal_(self.lookup_table)
        if padding:
            with torch.no_grad(): self.lookup_table[0].zero_()

    def forward(self,tokens):
        values = F.embedding(tokens,self.lookup_table,padding_idx=0 if self.padding else None)
        return values*math.sqrt(self.width) if self.scale else values


class Position(nn.Module):
    def __init__(self,width,dropout,maximum,*,learned=False):
        super().__init__()
        self.maximum,self.learned = maximum,learned
        self.dropout = nn.Dropout(dropout)
        if learned:
            self.pos_emb_table = Embedding(maximum,width,padding=False,scale=False)
            self.register_buffer('pos_vector',torch.arange(maximum))
        else:
            positions = torch.arange(maximum,dtype=torch.float32)[:,None]
            frequencies = torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000.)/width))
            values = torch.zeros(maximum,1,width)
            values[:,0,0::2] = (positions*frequencies).sin()
            values[:,0,1::2] = (positions*frequencies[:width//2]).cos()
            self.register_buffer('pe',values)

    def forward(self,values):
        if values.shape[1]>self.maximum:
            raise ValueError('GeoSAN sequence exceeds its positional capacity')
        positional = self.pos_emb_table(self.pos_vector[:values.shape[1]])[None] if self.learned else self.pe[:values.shape[1]].transpose(0,1)
        return self.dropout(values+positional)


class GeoSAN(nn.Module):
    def __init__(self,*,metadata,location_dim=50,region_dim=50,user_dim=0,time_dim=0,
                 num_heads=1,num_layers=2,dropout=.5,max_len=5000,candidate_chunk_size=256,
                 position_encoding='transformer',embedding_fusion='concat',use_location_only=False,
                 size_sqrt_regularize=True,use_attention_as_decoder=False):
        super().__init__()
        if (min(location_dim,region_dim,num_heads,num_layers,max_len,candidate_chunk_size)<1
                or min(user_dim,time_dim)<0 or not 0<=dropout<1
                or position_encoding not in ('transformer','embedding') or embedding_fusion not in ('concat','multiply')):
            raise ValueError('GeoSAN requires valid positive widths, heads, lengths, dropout and fusion/position choices')
        info = metadata.info
        if max_len<info['max_prefix_length']:
            raise ValueError(f'GeoSAN max_len must cover the longest prefix: {info["max_prefix_length"]}')
        width = location_dim if use_location_only else location_dim+region_dim
        if width%num_heads:
            raise ValueError('GeoSAN encoder width must be divisible by num_heads')
        self.max_len,self.chunk_size = max_len,candidate_chunk_size
        self.fusion,self.location_only = embedding_fusion,use_location_only
        self.regularize,self.attention_decoder = size_sqrt_regularize,use_attention_as_decoder
        self.user_dim,self.time_dim = user_dim,time_dim
        self.register_buffer('quadkey_codes',torch.tensor(info['quadkey_codes'].copy(),dtype=torch.long))
        self.emb_loc = Embedding(info['loc_size'],location_dim)
        components = [location_dim]
        if not use_location_only:
            self.emb_reg = Embedding(len(info['quadkey_tokens']),region_dim)
            self.region_pos_encoder = Position(region_dim,dropout,max(20,self.quadkey_codes.shape[1]),learned=True)
            layer = nn.TransformerEncoderLayer(region_dim,1,region_dim,dropout,batch_first=True)
            self.region_encoder = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            components.append(region_dim)
            if user_dim:
                self.emb_user = Embedding(info['uid_size'],user_dim);components.append(user_dim)
            if time_dim:
                self.emb_time = Embedding(info['tim_size'],time_dim);components.append(time_dim)
        if embedding_fusion=='multiply' and not use_location_only and len(set(components))!=1:
            raise ValueError('GeoSAN multiply fusion requires equal active embedding widths')
        incoming = sum(components) if embedding_fusion=='concat' and not use_location_only else location_dim
        self.source_projection = nn.Identity() if incoming==width else nn.Linear(incoming,width,bias=False)
        self.pos_encoder = Position(width,dropout,max_len,learned=position_encoding=='embedding')
        layer = nn.TransformerEncoderLayer(width,num_heads,width,dropout,batch_first=True)
        self.encoder = nn.TransformerEncoder(layer,num_layers,enable_nested_tensor=False)
        if use_attention_as_decoder:
            self.layer_norm = nn.LayerNorm(width)

    def region(self,locations):
        words = self.quadkey_codes[locations]
        encoded = self.emb_reg(words).reshape(-1,words.shape[-1],self.emb_reg.width)
        encoded = self.region_encoder(self.region_pos_encoder(encoded)).mean(1)
        return encoded.reshape(*locations.shape,self.emb_reg.width)

    def encode(self,batch):
        locations,times,lengths,users = (batch[key] for key in ('current_loc','current_tim','lengths','uid'))
        if (locations.ndim!=2 or times.shape!=locations.shape or lengths.shape!=(len(locations),)
                or users.shape!=(len(locations),) or not len(locations)
                or (lengths<1).any() or (lengths>locations.shape[1]).any() or (lengths>self.max_len).any()):
            raise ValueError('GeoSAN requires matching prefixes and positive lengths within max_len')
        stop = int(lengths.max())
        valid = torch.arange(stop,device=locations.device)[None]<lengths[:,None]
        locations,times = locations[:,:stop].masked_fill(~valid,0),times[:,:stop].masked_fill(~valid,0)
        if ((locations[valid]<1).any() or (locations>=self.emb_loc.lookup_table.shape[0]).any()
                or (times[valid]<1).any() or (times>168).any()):
            raise ValueError('GeoSAN observed location/time tokens are outside their vocabulary')
        parts = [self.emb_loc(locations)]
        if not self.location_only:
            parts.append(self.region(locations))
            if self.user_dim:
                if (users<1).any() or (users>=len(self.emb_user.lookup_table)).any():
                    raise ValueError('GeoSAN user tokens are outside their vocabulary')
                parts.append(self.emb_user(users)[:,None].expand(-1,stop,-1))
            if self.time_dim: parts.append(self.emb_time(times))
        fused = torch.cat(parts,dim=-1) if self.fusion=='concat' else torch.stack(parts).prod(0)
        source = self.source_projection(fused)
        if self.regularize: source = source*math.sqrt(source.shape[-1])
        source = self.pos_encoder(source)
        causal = torch.ones(stop,stop,dtype=torch.bool,device=source.device).triu(1)
        encoded = self.encoder(source,mask=causal,src_key_padding_mask=~valid).masked_fill(~valid[...,None],0.)
        last = encoded.gather(1,(lengths-1)[:,None,None].expand(-1,1,encoded.shape[-1])).squeeze(1)
        return encoded,valid,last

    def candidates(self,tokens):
        location = self.emb_loc(tokens)
        return location if self.location_only else torch.cat((location,self.region(tokens)),dim=-1)

    def score(self,encoded,valid,last,candidates):
        if (candidates.ndim not in (1,2) or not candidates.shape[-1] or (candidates<0).any()
                or (candidates>=len(self.quadkey_codes)-1).any()
                or (candidates.ndim==2 and len(candidates)!=len(last))):
            raise ValueError('GeoSAN candidates must be nonempty vocabulary indices, shared or per sample')
        scores = []
        for start in range(0,candidates.shape[-1],self.chunk_size):
            target = self.candidates(candidates[...,start:start+self.chunk_size]+1)
            if target.ndim==2: target = target[None].expand(len(last),-1,-1)
            output = last[:,None]
            if self.attention_decoder:
                logits = target@encoded.transpose(-1,-2)/math.sqrt(encoded.shape[-1])
                weights = logits.masked_fill(~valid[:,None],-torch.inf).softmax(-1)
                output = self.layer_norm(weights@encoded+output)
            scores.append((output*target).sum(-1))
        return torch.cat(scores,dim=-1)

    def forward(self,batch):
        return self.score(*self.encode(batch),torch.arange(len(self.quadkey_codes)-1,device=self.quadkey_codes.device))

    def forward_candidates(self,batch,candidates):
        return self.score(*self.encode(batch),candidates)
