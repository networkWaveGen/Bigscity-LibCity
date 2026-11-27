"""Three-expert routing returns forecasts; expert supervision is training-only."""
import torch
from torch import nn

from libcity.model.traffic_speed_prediction.TESTAM import TemporalModel,STModel,AttentionModel,MemoryGate


class AdaptiveExpert(STModel):
    def __init__(self,hidden,nodes,features,outputs,layers,dropout):
        super().__init__(hidden,2,nodes,dropout,layers,out_dim=1,in_dim=features)
        if outputs!=1:
            self.proj = nn.Linear(hidden,hidden+outputs)
        self.out_dim = outputs

    def forward(self,source,previous,supports):
        hidden = self.start_linear(source.permute(0,2,3,1))
        initial = hidden
        states = []
        for temporal,spatial,cross,feedforward in zip(self.temporal_layers,self.spatial_layers,self.ed_layers,self.ff):
            hidden = spatial(temporal(hidden),supports)
            if previous is not None:
                hidden = cross(hidden,previous[-1])
            hidden = feedforward(hidden)
            states.append(hidden)
        combined = self.proj(self.act(hidden))
        # The original projection contains a hidden residual plus prediction;
        # routing must use only the final output channels.
        return initial-combined[...,:initial.shape[-1]],combined[...,-self.out_dim:],states


class TESTAM(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,hidden_size=32,layers=3,dropout=0.,
                 prob_mul=False,is_quantile=True,quantile=.7):
        super().__init__()
        if hidden_size<4 or hidden_size%4 or layers<1 or not 0<=dropout<1 or not .5<=quantile<1:
            raise ValueError('TESTAM requires hidden_size divisible by 4, positive layers, valid dropout and quantile in [0.5,1)')
        if output_window>input_window or metadata.input_layout.get('kind')!='window':
            raise ValueError('TESTAM requires a contiguous window and output_window <= input_window')
        self.time_index = metadata.input_layout.get('calendar',{}).get('time_in_day')
        if self.time_index is None:
            raise ValueError('TESTAM requires data.features.time_in_day=true')
        interval = metadata.info.get('time_intervals',300)
        if interval<=0 or 86400%interval:
            raise ValueError('TESTAM requires time_intervals to divide one day')
        self.vocab_size,self.horizon = int(86400//interval),output_window
        self.output_dim,self.prob_mul = metadata.output_dim,prob_mul
        self.is_quantile,self.quantile = is_quantile,quantile
        self.observed = tuple(metadata.input_layout['observed_feature_indices'])
        future = tuple(metadata.input_layout.get('future_feature_indices',()))
        self.future_time_index = future.index(self.time_index) if self.time_index in future else None
        self.identity_expert = TemporalModel(hidden_size,metadata.num_nodes,layers,dropout,
            in_dim=len(self.observed),vocab_size=self.vocab_size,output_dim=metadata.output_dim)
        self.adaptive_expert = AdaptiveExpert(hidden_size,metadata.num_nodes,metadata.feature_dim,metadata.output_dim,layers,dropout)
        self.attention_expert = AttentionModel(hidden_size,layers,dropout,in_dim=metadata.feature_dim,out_dim=metadata.output_dim)
        self.gate_network = MemoryGate(hidden_size,metadata.num_nodes,input_dim=metadata.feature_dim)
        for expert in (self.identity_expert,self.adaptive_expert,self.attention_expert):
            for parameter in expert.parameters():
                if parameter.ndim>1:
                    nn.init.xavier_uniform_(parameter)

    def _experts(self,batch):
        source = batch['X'].transpose(1,3)
        current = (source[:,self.time_index,0]*self.vocab_size).round().long().remainder(self.vocab_size)
        future = (current+self.horizon).remainder(self.vocab_size)
        if 'future_features' in batch and self.future_time_index is not None:
            known = batch['future_features'][:,:,0,self.future_time_index]
            future = future.clone()
            future[:,-self.horizon:] = (known*self.vocab_size).round().long().remainder(self.vocab_size)
        first = self.gate_network.We1@self.gate_network.memory
        second = self.gate_network.We2@self.gate_network.memory
        supports = [torch.softmax(torch.relu(first@second.T),dim=-1),torch.softmax(torch.relu(second@first.T),dim=-1)]
        identity,identity_states = self.identity_expert(current,source[:,list(self.observed)])
        _,future_states = self.identity_expert(future)
        _,adaptive,adaptive_states = self.adaptive_expert(source,future_states,supports)
        attention,attention_state = self.attention_expert(source,future_states)
        experts = torch.stack((identity,adaptive,attention),dim=-2)
        gate = torch.softmax(self.gate_network(source.permute(0,2,3,1),
            [identity_states[-1],adaptive_states[-1],attention_state]),dim=-1)
        chosen = gate.argmax(dim=-1)
        selection = chosen[...,None,None].expand(*chosen.shape,1,self.output_dim)
        prediction = experts.gather(-2,selection).squeeze(-2)
        if self.prob_mul:
            prediction = prediction*gate.max(dim=-1).values.unsqueeze(-1)
        return (prediction[:,:,-self.horizon:].permute(0,2,1,3),
                gate[:,:,-self.horizon:],experts[:,:,-self.horizon:])

    def forward(self,batch):
        return self._experts(batch)[0]

    def forward_with_aux(self,batch):
        return self._experts(batch)
