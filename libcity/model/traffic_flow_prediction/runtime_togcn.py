"""Graph/LSTM encoding with an explicit training-only teacher path."""
import random

import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.traffic_flow_prediction.ToGCN import GCN


class Encoder(nn.Module):
    def __init__(self,nodes,features,hidden,layers,dropout,graph_hidden):
        super().__init__()
        self.gcn = GCN(features,graph_hidden,1,torch.device('cpu'))
        self.lstm = nn.LSTM(nodes,hidden,num_layers=layers,batch_first=True,
                            dropout=dropout if layers>1 else 0.)

    def forward(self,source,graph,state):
        encoded = self.gcn(source,graph).reshape(source.shape[0],1,-1)
        return self.lstm(encoded,state)[1]


class Decoder(nn.Module):
    def __init__(self,nodes,outputs,hidden,layers,dropout):
        super().__init__()
        self.lstm = nn.LSTM(nodes*outputs,hidden,num_layers=layers,batch_first=True,
                            dropout=dropout if layers>1 else 0.)
        self.dense = nn.Linear(hidden,nodes*outputs)

    def forward(self,source,state):
        decoded,state = self.lstm(source.reshape(source.shape[0],1,-1),state)
        return F.relu(self.dense(decoded[:,0])),state


class ToGCN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,hidden_size=128,decoder_t=3,
                 teacher_forcing_ratio=.5,num_layers=2,dropout=.5,graph_hidden=128):
        super().__init__()
        if min(hidden_size,num_layers,graph_hidden)<1 or not 0<=dropout<1 or not 0<=teacher_forcing_ratio<=1:
            raise ValueError('ToGCN requires positive widths/layers and valid dropout/teacher forcing')
        if not 0<=decoder_t<input_window or metadata.input_layout.get('kind')!='window':
            raise ValueError('ToGCN requires contiguous input and 0 <= decoder_t < input_window')
        self.register_buffer('adjacency',torch.tensor(metadata.adjacency.copy(),dtype=torch.float32))
        self.nodes,self.outputs = metadata.num_nodes,metadata.output_dim
        self.horizon,self.warmup = output_window,decoder_t
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.future_indices = tuple(metadata.input_layout.get('future_feature_indices',()))
        self.encoder = Encoder(self.nodes,metadata.feature_dim,hidden_size,num_layers,dropout,graph_hidden)
        self.decoder = Decoder(self.nodes,self.outputs,hidden_size,num_layers,dropout)
        self.linear = nn.Linear(metadata.feature_dim,self.outputs)

    def _decode(self,batch,targets=None):
        source = batch['X']
        state = None
        for position in range(source.shape[1]):
            state = self.encoder(source[:,position],self.adjacency,state)
        for position in range(source.shape[1]-self.warmup-1,source.shape[1]-1):
            _,state = self.decoder(self.linear(source[:,position]),state)
        following = self.linear(source[:,-1])
        use_teacher = targets is not None and random.random()<self.teacher_forcing_ratio
        predictions = []
        for position in range(self.horizon):
            prediction,state = self.decoder(following,state)
            prediction = prediction.reshape(source.shape[0],self.nodes,self.outputs)
            predictions.append(prediction)
            following = prediction
            if use_teacher:
                # Only the target and declared future channels may change.
                # Unknown future observations persist from the last history.
                teacher = source[:,-1].clone()
                teacher[...,:self.outputs] = targets[:,position,:,:self.outputs]
                if self.future_indices:
                    if 'future_features' not in batch:
                        raise ValueError('ToGCN teacher forcing requires explicit known future features')
                    teacher[...,list(self.future_indices)] = batch['future_features'][:,position]
                following = self.linear(teacher)
        return torch.stack(predictions,dim=1)

    def forward(self,batch):
        return self._decode(batch)

    def forward_with_teacher(self,batch,targets):
        return self._decode(batch,targets)
