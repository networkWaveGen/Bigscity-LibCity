"""Memory-augmented recurrent prediction with training-only curriculum inputs."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.executor.evaluate import StepResult
from libcity.model.traffic_speed_prediction.MegaCRN import ADCRNN_Encoder, ADCRNN_Decoder


class MegaCRN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,rnn_units=64,num_layers=1,cheb_k=3,
                 cl_decay_steps=2000,use_curriculum_learning=True,mem_num=20,mem_dim=64,lamb=.01,lamb1=.01):
        super().__init__()
        if min(rnn_units,num_layers,mem_dim,cl_decay_steps)<1 or cheb_k<2 or mem_num<2 or min(lamb,lamb1)<0:
            raise ValueError('MegaCRN requires positive dimensions/decay, cheb_k>=2, mem_num>=2 and nonnegative auxiliary weights')
        self.num_nodes,self.output_dim,self.horizon = metadata.num_nodes,metadata.output_dim,output_window
        self.rnn_units,self.num_layers = rnn_units,num_layers
        self.ycov_dim = len(metadata.input_layout.get('future_feature_indices',()))
        self.cl_decay_steps,self.use_curriculum_learning = cl_decay_steps,use_curriculum_learning
        self.lamb,self.lamb1 = lamb,lamb1
        self.memory = nn.ParameterDict({
            'Memory':nn.Parameter(torch.randn(mem_num,mem_dim)),
            'Wq':nn.Parameter(torch.randn(rnn_units,mem_dim)),
            'We1':nn.Parameter(torch.randn(metadata.num_nodes,mem_num)),
            'We2':nn.Parameter(torch.randn(metadata.num_nodes,mem_num)),
        })
        for parameter in self.memory.values():
            nn.init.xavier_normal_(parameter)
        self.encoder = ADCRNN_Encoder(metadata.num_nodes,metadata.output_dim,rnn_units,cheb_k,num_layers)
        self.decoder = ADCRNN_Decoder(metadata.num_nodes,metadata.output_dim+self.ycov_dim,rnn_units+mem_dim,cheb_k,num_layers)
        self.proj = nn.Sequential(nn.Linear(rnn_units+mem_dim,metadata.output_dim))

    def compute_sampling_threshold(self,batches_seen):
        exponent = batches_seen/self.cl_decay_steps
        return 0. if exponent>700 else self.cl_decay_steps/(self.cl_decay_steps+math.exp(exponent))

    def query_memory(self,hidden):
        query = hidden@self.memory['Wq']
        attention = torch.softmax(query@self.memory['Memory'].T,dim=-1)
        value = attention@self.memory['Memory']
        indices = attention.topk(2,dim=-1).indices
        positive = self.memory['Memory'][indices[...,0]]
        negative = self.memory['Memory'][indices[...,1]]
        return value,query,positive,negative

    def _forecast(self,batch,targets=None,batches_seen=0):
        source = batch['X'][...,:self.output_dim]
        first = self.memory['We1']@self.memory['Memory']
        second = self.memory['We2']@self.memory['Memory']
        supports = [torch.softmax(F.relu(first@second.T),dim=-1),torch.softmax(F.relu(second@first.T),dim=-1)]
        initial = [source.new_zeros(source.shape[0],self.num_nodes,self.rnn_units) for _ in range(self.num_layers)]
        encoded,_ = self.encoder(source,initial,supports)
        hidden = encoded[:,-1]
        value,query,positive,negative = self.query_memory(hidden)
        state = [torch.cat((hidden,value),dim=-1)]*self.num_layers
        following = source.new_zeros(source.shape[0],self.num_nodes,self.output_dim)
        if self.ycov_dim:
            if 'future_features' not in batch:
                raise ValueError('MegaCRN requires explicit known future_features, never target-packed covariates')
            covariates = batch['future_features']
        else:
            covariates = source.new_zeros(source.shape[0],self.horizon,self.num_nodes,0)
        predictions = []
        for index in range(self.horizon):
            decoded,state = self.decoder(torch.cat((following,covariates[:,index]),dim=-1),state,supports)
            following = self.proj(decoded)
            predictions.append(following)
            if self.training and self.use_curriculum_learning and targets is not None:
                if np.random.uniform(0,1)<self.compute_sampling_threshold(batches_seen):
                    following = targets[:,index,...,:self.output_dim]
        return torch.stack(predictions,dim=1),value,query,positive,negative

    def forward(self,batch):
        return self._forecast(batch)[0]

    def forward_with_aux(self,batch,*,targets=None,batches_seen=0):
        return self._forecast(batch,targets,batches_seen)


def memory_step(loss_fn):
    def step(model,batch,state):
        prediction,_,query,positive,negative = model.forward_with_aux(loss_fn.inputs(batch),
            targets=batch['y'],batches_seen=state.batches_seen)
        prediction_loss = loss_fn(prediction,batch['y'],batch.get('y_raw'))
        separation = F.triplet_margin_loss(query,positive.detach(),negative.detach(),margin=1.)
        compactness = F.mse_loss(query,positive.detach())
        return StepResult(prediction_loss.loss+model.lamb*separation+model.lamb1*compactness,prediction_loss.weight)
    return step
