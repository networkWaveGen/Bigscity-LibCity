"""Graph structure learning and diffusion GRU with a deterministic inference path."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_norm import SingletonBatchNorm1d


def sampled_graph(logits, temperature, uniforms=None):
    if type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('GTS Gumbel temperature must be finite and positive')
    scores = logits.float() if logits.dtype in (torch.float16,torch.bfloat16) else logits
    uniforms = torch.rand_like(scores) if uniforms is None else uniforms.to(scores)
    limits = torch.finfo(scores.dtype)
    noise = -torch.log(-torch.log(uniforms.clamp(limits.tiny,1-limits.eps)))
    perturbed = scores+noise
    soft = ((perturbed-perturbed.max(-1,keepdim=True).values)/temperature).softmax(-1)
    hard = F.one_hot(soft.argmax(-1),2).to(soft.dtype)
    return (hard-soft.detach()+soft)[...,0]


class DiffusionLinear(nn.Module):
    def __init__(self, inputs, units, outputs, depth, bias):
        super().__init__()
        self.depth = depth
        self.weight = nn.Parameter(torch.empty((inputs+units)*(depth+1),outputs))
        self.biases = nn.Parameter(torch.empty(outputs))
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.biases,bias)

    def forward(self, inputs, hidden, support):
        value = torch.cat((inputs,hidden),-1)
        terms = [value]
        if self.depth:
            terms.append(torch.einsum('nm,bmc->bnc',support,value))
        for _ in range(2,self.depth+1):
            terms.append(2*torch.einsum('nm,bmc->bnc',support,terms[-1])-terms[-2])
        return torch.stack(terms,-1).flatten(2)@self.weight+self.biases


class DiffusionCell(nn.Module):
    def __init__(self, inputs, units, depth):
        super().__init__()
        self._fn = DiffusionLinear(inputs,units,2*units,depth,1.)
        self._gconv = DiffusionLinear(inputs,units,units,depth,0.)

    def forward(self, inputs, hidden, support):
        reset,update = self._fn(inputs,hidden,support).sigmoid().chunk(2,-1)
        candidate = self._gconv(inputs,reset*hidden,support).tanh()
        return update*hidden+(1-update)*candidate


class DiffusionStack(nn.Module):
    def __init__(self, inputs, units, layers, depth, outputs=None):
        super().__init__()
        self.units = units
        if outputs is not None:
            self.projection_layer = nn.Linear(units,outputs)
        self.dcgru_layers = nn.ModuleList(DiffusionCell(inputs if index==0 else units,units,depth) for index in range(layers))

    def forward(self, inputs, support, hidden=None):
        if hidden is None:
            hidden = inputs.new_zeros(len(self.dcgru_layers),inputs.shape[0],inputs.shape[1],self.units)
        states = []
        for index,cell in enumerate(self.dcgru_layers):
            inputs = cell(inputs,hidden[index],support)
            states.append(inputs)
        if hasattr(self,'projection_layer'):
            inputs = self.projection_layer(inputs)
        return inputs,torch.stack(states)


class GTS(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, rnn_units=64, num_rnn_layers=1,
                 max_diffusion_step=3, kernel_size=10, embedding_dim=100):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,rnn_units=rnn_units,
                num_rnn_layers=num_rnn_layers,kernel_size=kernel_size,embedding_dim=embedding_dim,
                num_nodes=metadata.num_nodes,feature_dim=metadata.feature_dim,output_dim=metadata.output_dim).items():
            if type(value) is not int or value<1:
                raise ValueError(f'GTS {name} must be a positive integer')
        if type(max_diffusion_step) is not int or max_diffusion_step<0:
            raise ValueError('GTS max_diffusion_step must be a nonnegative integer')
        if metadata.input_layout.get('kind')!='window' or 'graph_history' not in metadata.static_data:
            raise ValueError('GTS requires static training graph_history and continuous windows')
        history = np.array(metadata.static_data['graph_history'],dtype=np.float32,copy=True)
        if history.ndim!=2 or history.shape[1]!=metadata.num_nodes or not np.isfinite(history).all() or len(history)<2*kernel_size-1:
            raise ValueError('GTS graph_history must be finite [training_steps,nodes] with at least 2*kernel_size-1 steps')
        prior = np.array(metadata.adjacency,dtype=np.float32,copy=True)
        if prior.shape!=(metadata.num_nodes,metadata.num_nodes) or not np.isfinite(prior).all() or np.any((prior<0)|(prior>1)):
            raise ValueError('GTS prior adjacency must match nodes and contain probabilities in [0,1]')
        self.nodes,self.features,self.output_dim = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.input_window,self.output_window = input_window,output_window
        self.register_buffer('node_features',torch.from_numpy(history))
        self.register_buffer('graph_prior',torch.from_numpy(prior))
        self.register_buffer('identity',torch.eye(self.nodes))
        self.register_buffer('off_diagonal',~torch.eye(self.nodes,dtype=torch.bool))
        self.encoder_model = DiffusionStack(self.features,rnn_units,num_rnn_layers,max_diffusion_step)
        self.decoder_model = DiffusionStack(self.output_dim,rnn_units,num_rnn_layers,max_diffusion_step,self.output_dim)
        self.conv1 = nn.Conv1d(1,8,kernel_size)
        self.conv2 = nn.Conv1d(8,16,kernel_size)
        self.fc = nn.Linear((len(history)-2*kernel_size+2)*16,embedding_dim)
        self.bn1,self.bn2,self.bn3 = SingletonBatchNorm1d(8),SingletonBatchNorm1d(16),SingletonBatchNorm1d(embedding_dim)
        self.fc_out = nn.Linear(2*embedding_dim,embedding_dim)
        self.fc_cat = nn.Linear(embedding_dim,2)

    def graph_logits(self):
        value = self.node_features.T[:,None]
        value = self.bn1(self.conv1(value).relu())
        value = self.bn2(self.conv2(value).relu())
        value = self.bn3(self.fc(value.flatten(1)).relu())
        width = value.shape[-1]
        # Linear([sender,receiver]) factors before ReLU without N^2-by-N
        # incidence matrices or repeated per-edge dense projections.
        senders = F.linear(value,self.fc_out.weight[:,:width])
        receivers = F.linear(value,self.fc_out.weight[:,width:],self.fc_out.bias)
        return self.fc_cat((receivers[:,None]+senders[None]).relu())

    def forecast(self, source, adjacency, targets=None, teacher_mask=None):
        if source.ndim!=4 or source.shape[1:]!=(self.input_window,self.nodes,self.features):
            raise ValueError('GTS X must have configured [batch,history,node,feature] dimensions')
        if targets is not None and targets.shape!=(len(source),self.output_window,self.nodes,self.output_dim):
            raise ValueError('GTS teacher targets must match [batch,horizon,node,target] dimensions')
        value = adjacency+self.identity
        support = (value/value.sum(-1,keepdim=True)).T
        hidden = None
        for inputs in source.unbind(1):
            _,hidden = self.encoder_model(inputs,support,hidden)
        inputs = source.new_zeros(len(source),self.nodes,self.output_dim)
        predictions = []
        for index in range(self.output_window):
            inputs,hidden = self.decoder_model(inputs,support,hidden)
            predictions.append(inputs)
            if targets is not None and teacher_mask is not None and index+1<self.output_window and teacher_mask[index]:
                inputs = targets[:,index]
        return torch.stack(predictions,1)

    def inference_graph(self):
        logits = self.graph_logits()
        return (logits.argmax(-1)==0).to(logits.dtype)*self.off_diagonal

    def forward(self, inputs):
        return self.forecast(inputs['X'],self.inference_graph())

    def forward_with_aux(self, inputs, *, targets=None, teacher_mask=None, temperature=.5, uniforms=None):
        logits = self.graph_logits()
        adjacency = sampled_graph(logits,temperature,uniforms)*self.off_diagonal
        return self.forecast(inputs['X'],adjacency,targets,teacher_mask),logits[...,0]
