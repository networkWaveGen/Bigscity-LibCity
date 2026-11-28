"""Mixed explicit/implicit graph recurrence with explicit train-time sampling."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_graph import chebyshev_basis,scaled_laplacian


def sample_edges(logits,temperature=1.,noise=None):
    if type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('RGSL graph temperature must be finite and positive')
    values = logits.float() if logits.dtype in (torch.float16,torch.bfloat16) else logits
    noise = -torch.empty_like(values).exponential_().clamp_min(torch.finfo(values.dtype).tiny).log() if noise is None else noise.to(values)
    scores = (values+noise)/temperature
    hard = F.one_hot(scores.argmax(-1),2).to(scores.dtype)
    # RGSL's author implementation uses the perturbed score here,
    # without a softmax. Preserve this linear straight-through derivative.
    return (hard+(scores-scores.detach()))[...,0]


class AttentionGate(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channels,max(1,channels//16),bias=False),nn.ReLU(),
            nn.Linear(max(1,channels//16),1,bias=False),nn.Hardsigmoid())

    def forward(self,values):
        return values*self.fc(values.mean(-1)).unsqueeze(-1)


class MixedGraph(nn.Module):
    def __init__(self,inputs,outputs,order,embedding):
        super().__init__()
        self.weights_pool = nn.Parameter(torch.empty(embedding,order,inputs,outputs))
        self.bias_pool = nn.Parameter(torch.empty(embedding,outputs))
        self.init_gconv = nn.Linear(inputs,outputs)
        self.gconv = nn.Linear(outputs*order,outputs)
        self.dy_gate1,self.dy_gate2 = AttentionGate(outputs),AttentionGate(outputs)

    def prepare(self,embeddings):
        return torch.einsum('nd,dkio->nkio',embeddings,self.weights_pool),embeddings@self.bias_pool

    def forward(self,values,learned_basis,static_basis,parameters):
        weight,bias = parameters
        propagated = torch.einsum('knm,bmc->bnkc',learned_basis,values)
        implicit = torch.einsum('bnki,nkio->bno',propagated,weight)+bias
        explicit = torch.einsum('knm,bmc->bnkc',static_basis,self.init_gconv(values)).flatten(2)
        explicit = self.gconv(explicit)
        return (self.dy_gate1(F.leaky_relu(implicit).transpose(1,2))+
            self.dy_gate2(F.leaky_relu(explicit).transpose(1,2))).transpose(1,2)


class Cell(nn.Module):
    def __init__(self,inputs,units,order,embedding):
        super().__init__()
        self.gate = MixedGraph(inputs+units,2*units,order,embedding)
        self.update = MixedGraph(inputs+units,units,order,embedding)

    def forward(self,values,state,learned_basis,static_basis,parameters):
        gate,update = parameters
        reset,keep = self.gate(torch.cat((values,state),-1),learned_basis,static_basis,gate).sigmoid().chunk(2,-1)
        candidate = self.update(torch.cat((values,reset*state),-1),learned_basis,static_basis,update).tanh()
        return keep*state+(1-keep)*candidate


class Encoder(nn.Module):
    def __init__(self,inputs,units,layers,order,embedding):
        super().__init__()
        self.units = units
        self.dcrnn_cells = nn.ModuleList(Cell(inputs if index==0 else units,units,order,embedding) for index in range(layers))

    def forward(self,source,embeddings,learned_basis,static_basis):
        for index,cell in enumerate(self.dcrnn_cells):
            # Embedding-dependent weights and graph polynomials do not vary
            # over the observed time axis. Prepare them once per forward.
            parameters = cell.gate.prepare(embeddings),cell.update.prepare(embeddings)
            state = source.new_zeros(source.shape[0],source.shape[2],self.units)
            sequences = []
            last = index+1==len(self.dcrnn_cells)
            for values in source.unbind(1):
                state = cell(values,state,learned_basis,static_basis,parameters)
                if not last:
                    sequences.append(state)
            if not last:
                source = torch.stack(sequences,1)
        return state


class RGSL(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,rnn_units=64,num_layers=2,embed_dim=8,cheb_k=2):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,rnn_units=rnn_units,
                num_layers=num_layers,embed_dim=embed_dim,cheb_k=cheb_k,num_nodes=metadata.num_nodes,
                feature_dim=metadata.feature_dim,output_dim=metadata.output_dim).items():
            if type(value) is not int or value<1:
                raise ValueError(f'RGSL {name} must be a positive integer')
        self.nodes,self.features,self.outputs = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.input_window,self.output_window,self.order = input_window,output_window,cheb_k
        basis = chebyshev_basis(scaled_laplacian(metadata.adjacency),cheb_k)
        if basis.shape[1:]!=(self.nodes,self.nodes):
            raise ValueError('RGSL graph shape must match metadata nodes')
        self.register_buffer('static_basis',basis)
        self.register_buffer('identity',torch.eye(self.nodes))
        self.register_buffer('off_diagonal',~torch.eye(self.nodes,dtype=torch.bool))
        self.node_embeddings = nn.Parameter(torch.empty(self.nodes,embed_dim))
        self.encoder = Encoder(self.features,rnn_units,num_layers,cheb_k,embed_dim)
        self.end_conv = nn.Conv2d(1,output_window*self.outputs,(1,rnn_units))
        for value in self.parameters():
            if value.ndim>1:
                nn.init.xavier_uniform_(value)
            else:
                nn.init.uniform_(value)

    def edge_logits(self):
        embeddings = self.node_embeddings.float() if self.node_embeddings.dtype in (torch.float16,torch.bfloat16) else self.node_embeddings
        normalized = F.normalize(embeddings,dim=-1,eps=1e-12)
        affinity = ((normalized@normalized.T).clamp(-1,1)+1)/2
        return torch.stack((affinity,1-affinity),-1)

    def learned_basis(self,adjacency):
        graph = adjacency*self.off_diagonal
        laplacian = torch.diag_embed(graph.sum(-1))-graph
        scale = laplacian.amax()-laplacian.amin()
        scale = torch.where(scale>0,scale,torch.ones_like(scale))
        return chebyshev_basis(2*laplacian/scale-self.identity,self.order)

    def forecast(self,source,learned_basis):
        if source.ndim!=4 or source.shape[1:]!=(self.input_window,self.nodes,self.features):
            raise ValueError('RGSL X must match configured [batch,history,node,feature] dimensions')
        hidden = self.encoder(source,self.node_embeddings,learned_basis,self.static_basis)
        prediction = self.end_conv(hidden[:,None]).squeeze(-1)
        return prediction.reshape(len(source),self.output_window,self.outputs,self.nodes).transpose(2,3)

    def inference_basis(self):
        logits = self.edge_logits()
        return self.learned_basis((logits.argmax(-1)==0).to(logits.dtype))

    def forward(self,inputs):
        return self.forecast(inputs['X'],self.inference_basis())

    def forward_with_aux(self,inputs,*,temperature=1.,noise=None):
        adjacency = sample_edges(self.edge_logits(),temperature,noise)*self.off_diagonal
        return self.forecast(inputs['X'],self.learned_basis(adjacency)),adjacency
