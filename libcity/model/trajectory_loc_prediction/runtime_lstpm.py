"""Long/short-term session attention and a causal spatially dilated LSTM."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


def attend(sessions,query,valid,distances=None):
    scores = (sessions*query[:,None]).sum(-1)
    if distances is not None:
        scores = scores/distances
    scores = scores.masked_fill(~valid,-torch.inf)
    scores = torch.where(valid.any(1,keepdim=True),scores,0.)
    weights = scores.softmax(1).masked_fill(~valid,0.)
    return (sessions*weights[...,None]).sum(1)


class LSTPM(nn.Module):
    def __init__(self,*,metadata,emb_size=500,hidden_size=500):
        super().__init__()
        if min(emb_size,hidden_size)<1:
            raise ValueError('LSTPM requires positive embedding and hidden widths')
        info = metadata.info
        similarity = torch.tensor([list(row) for row in info['tim_sim_matrix']],dtype=torch.float32)
        if (similarity.shape!=(49,49) or not torch.isfinite(similarity).all()
                or (similarity<0).any() or (similarity>1).any()):
            raise ValueError('LSTPM requires a finite 49 x 49 time similarity matrix in [0,1]')
        self.register_buffer('tim_sim_matrix',similarity)
        self.hidden_size = hidden_size
        self.loc_emb = nn.Embedding(info['loc_size'],emb_size,padding_idx=0)
        self.lstmcell = nn.LSTM(emb_size,hidden_size,batch_first=True)
        self.lstmcell_history = nn.LSTM(emb_size,hidden_size,batch_first=True)
        self.dilated_rnn = nn.LSTMCell(emb_size,hidden_size)
        self.linear1 = nn.Linear(hidden_size,hidden_size,bias=False)
        self.linear = nn.Linear(2*hidden_size,info['loc_size'],bias=False)
        for name,parameter in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(parameter)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(parameter)
            elif 'bias' in name:
                nn.init.zeros_(parameter)

    def sequence(self,locations,lengths,recurrent,*,allow_empty=False):
        if locations.ndim!=2 or lengths.shape!=(len(locations),) or len(locations)==0:
            raise ValueError('LSTPM tokens and lengths must have matching nonempty batch shapes')
        if (lengths<(0 if allow_empty else 1)).any() or (lengths>locations.shape[1]).any():
            raise ValueError('LSTPM lengths must fit each padded sequence')
        if locations.shape[1]==0:
            locations = locations.new_zeros((len(locations),1))
        valid = torch.arange(locations.shape[1],device=locations.device)[None]<lengths[:,None]
        items = self.loc_emb(locations.masked_fill(~valid,0))
        packed = pack_padded_sequence(items,lengths.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
        values,_ = recurrent(packed)
        values,_ = pad_packed_sequence(values,batch_first=True,total_length=locations.shape[1])
        return items,values.masked_fill(~valid[...,None],0.),valid

    def dilated(self,items,parents,lengths,valid):
        if parents.shape!=valid.shape:
            raise ValueError('LSTPM spatial parents must match the current sequence')
        positions = torch.arange(parents.shape[1],device=parents.device)
        if (((parents<0)|(parents>=positions.clamp_min(1)[None]))&valid).any():
            raise ValueError('LSTPM spatial parents must index strictly earlier observations (first=0)')
        parents = parents.masked_fill(~valid,0)
        hidden,memory = [],[]
        for position in range(int(lengths.max())):
            if position==0:
                previous = items.new_zeros(len(items),self.hidden_size)
                state = (previous,previous)
            else:
                index = parents[:,position,None,None].expand(-1,1,self.hidden_size)
                # Retain immutable autograd states: mutating a gathered bank breaks backward.
                state = tuple(torch.stack(bank,1).gather(1,index).squeeze(1) for bank in (hidden,memory))
            h,c = self.dilated_rnn(items[:,position],state)
            active = valid[:,position,None]
            hidden.append(h.masked_fill(~active,0.));memory.append(c.masked_fill(~active,0.))
        index = (lengths-1)[:,None,None].expand(-1,1,self.hidden_size)
        return torch.stack(hidden,1).gather(1,index).squeeze(1)

    def long_context(self,batch,current_time,mean,last):
        lengths = batch['history_lengths']
        boundaries = batch['history_session_lengths']
        distances = batch['history_avg_distance']
        if (boundaries.ndim!=2 or boundaries.shape[0]!=len(last) or distances.shape!=boundaries.shape
                or (boundaries<0).any() or not torch.equal(boundaries.sum(1),lengths)):
            raise ValueError('LSTPM session lengths must be nonnegative and sum to history lengths')
        valid_sessions = boundaries>0
        if (valid_sessions[:,1:]&~valid_sessions[:,:-1]).any():
            raise ValueError('LSTPM session lengths must use trailing zero padding')
        if ((~torch.isfinite(distances)|(distances<1))&valid_sessions).any():
            raise ValueError('LSTPM valid history distances must be finite and at least one kilometre')
        _,history,valid = self.sequence(batch['history_loc'],lengths,self.lstmcell_history,allow_empty=True)
        if batch['history_tim'].shape!=batch['history_loc'].shape:
            raise ValueError('LSTPM history location and time shapes must match')
        if boundaries.shape[1]==0:
            return last.new_zeros(last.shape)
        times = batch['history_tim']
        if times.shape[1]==0:
            times = times.new_full((len(times),1),48)
        times = times.masked_fill(~valid,48)
        if ((times<0)|(times>48)|((times==48)&valid)).any():
            raise ValueError('LSTPM valid history time codes must lie in [0,47]')
        positions = torch.arange(history.shape[1],device=history.device).expand(len(history),-1).contiguous()
        session = torch.searchsorted(boundaries.cumsum(1).contiguous(),positions,right=True).masked_fill(~valid,0)
        weights = self.tim_sim_matrix[current_time[:,None],times].to(history.dtype).exp().masked_fill(~valid,0.)
        denominator = history.new_zeros(boundaries.shape).scatter_add(1,session,weights)
        pooled = history.new_zeros(*boundaries.shape,self.hidden_size).scatter_add(1,
            session[...,None].expand_as(history),history*weights[...,None])
        pooled = pooled/denominator.clamp_min(1.)[...,None]
        first = torch.selu(self.linear1(attend(pooled,mean,valid_sessions)))
        query = .5*(first+last)
        distances = distances.masked_fill(~valid_sessions,1.).to(history.dtype)
        return torch.selu(attend(pooled,query,valid_sessions,distances))

    def forward(self,batch):
        lengths = batch['lengths']
        items,current,valid = self.sequence(batch['current_loc'],lengths,self.lstmcell)
        if batch['current_tim'].shape!=batch['current_loc'].shape:
            raise ValueError('LSTPM current location and time shapes must match')
        times = batch['current_tim'].masked_fill(~valid,48)
        if (((times<0)|(times>47))&valid).any():
            raise ValueError('LSTPM valid current time codes must lie in [0,47]')
        index = (lengths-1)[:,None,None].expand(-1,1,self.hidden_size)
        last = current.gather(1,index).squeeze(1)
        mean = current.sum(1)/lengths[:,None]
        current_time = times.gather(1,(lengths-1)[:,None]).squeeze(1)
        long = self.long_context(batch,current_time,mean,last)
        spatial = self.dilated(items,batch['dilated_rnn_input_index'],lengths,valid)
        short = .5*(torch.selu(last)+torch.selu(spatial))
        return self.linear(torch.cat((long,short),dim=-1))[:,1:].log_softmax(-1)
