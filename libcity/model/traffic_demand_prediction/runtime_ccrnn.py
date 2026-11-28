"""Coupled evolving graph convolution with label-free sequence forecasting."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GraphConv(nn.Module):
    def __init__(self,inputs,outputs,depth):
        super().__init__()
        self.depth = depth
        self.out = nn.Linear(inputs*(depth+1),outputs)

    def forward(self,values,support):
        terms = [values]
        if self.depth:
            terms.append(torch.einsum('nm,bmc->bnc',support,values))
        for _ in range(2,self.depth+1):
            terms.append(2*torch.einsum('nm,bmc->bnc',support,terms[-1])-terms[-2])
        return self.out(torch.stack(terms,-1).flatten(2))


class EvolutionCell(nn.Module):
    def __init__(self,inputs,outputs,nodes,depth,layers):
        super().__init__()
        self.attlinear = nn.Linear(nodes*outputs,1) if layers>1 else None
        self.graphconv = nn.ModuleList(GraphConv(inputs if index==0 else outputs,outputs,depth) for index in range(layers))

    def forward(self,values,supports):
        outputs = []
        for layer,support in zip(self.graphconv,supports,strict=True):
            values = layer(values,support)
            outputs.append(values)
        if len(outputs)==1:
            return outputs[0]
        stacked = torch.stack(outputs,1)
        flat = stacked.flatten(2)
        attention = self.attlinear(flat).softmax(1)
        return (flat*attention).sum(1).reshape_as(values)


class Cell(nn.Module):
    def __init__(self,inputs,units,nodes,depth,layers):
        super().__init__()
        self.ru_gate_g_conv = EvolutionCell(inputs+units,2*units,nodes,depth,layers)
        self.candidate_g_conv = EvolutionCell(inputs+units,units,nodes,depth,layers)

    def forward(self,values,state,graphs):
        reset,keep = self.ru_gate_g_conv(torch.cat((values,state),-1),graphs).sigmoid().chunk(2,-1)
        candidate = self.candidate_g_conv(torch.cat((values,reset*state),-1),graphs).tanh()
        return keep*state+(1-keep)*candidate


class Encoder(nn.ModuleList):
    def __init__(self,inputs,units,nodes,depth,layers,graph_layers):
        super().__init__(Cell(inputs if index==0 else units,units,nodes,depth,graph_layers) for index in range(layers))
        self.units = units

    def forward(self,source,graphs):
        states = []
        for index,cell in enumerate(self):
            state = source.new_zeros(source.shape[0],source.shape[2],self.units)
            sequence = []
            last = index+1==len(self)
            for values in source.unbind(1):
                state = cell(values,state,graphs)
                if not last:
                    sequence.append(state)
            states.append(state)
            if not last:
                source = torch.stack(sequence,1)
        return states


class Decoder(nn.ModuleList):
    def __init__(self,outputs,units,nodes,depth,layers,graph_layers):
        super().__init__(Cell(outputs if index==0 else units,units,nodes,depth,graph_layers) for index in range(layers))
        self.out = nn.Linear(units,outputs)
        self.layers = layers

    def forward(self,source,states,graphs,horizon,outputs,targets=None,teacher_mask=None):
        values = source.new_zeros(len(source),source.shape[2],outputs)
        predictions = []
        for step in range(horizon):
            for index in range(self.layers):
                cell = self[index]
                values = cell(values,states[index],graphs)
                states[index] = values
            values = self.out(values)
            predictions.append(values)
            if targets is not None and teacher_mask is not None and step+1<horizon and teacher_mask[step]:
                values = targets[:,step]
        return torch.stack(predictions,1)


class CCRNN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,hidden_size=25,n_dim=50,k_hop=3,
                 n_rnn_layers=1,n_gconv_layers=3):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,hidden_size=hidden_size,
                n_dim=n_dim,n_rnn_layers=n_rnn_layers,n_gconv_layers=n_gconv_layers,num_nodes=metadata.num_nodes,
                feature_dim=metadata.feature_dim,output_dim=metadata.output_dim).items():
            if type(value) is not int or value<1:
                raise ValueError(f'CCRNN {name} must be a positive integer')
        if type(k_hop) is not int or k_hop<0:
            raise ValueError('CCRNN k_hop must be a nonnegative integer')
        support = np.array(metadata.adjacency,dtype=np.float32,copy=True)
        if support.shape!=(metadata.num_nodes,metadata.num_nodes) or not np.isfinite(support).all():
            raise ValueError('CCRNN initialization support must be a finite node-square matrix')
        self.nodes,self.features,self.outputs = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.input_window,self.output_window,self.depth = input_window,output_window,k_hop
        self.graph_layers,self.graph_dim = n_gconv_layers,min(n_dim,self.nodes)
        if k_hop:
            left,singular,right = torch.linalg.svd(torch.from_numpy(support),full_matrices=False)
            left,right = left[:,:self.graph_dim],right[:self.graph_dim]
            signs = left.gather(0,left.abs().argmax(0,keepdim=True)).sign()
            signs = torch.where(signs==0,torch.ones_like(signs),signs)
            left,right = left*signs,right*signs.T
            root = singular[:self.graph_dim].sqrt()
            # Keep the zero product while allowing a zero singular direction
            # to learn: only its right factor starts at zero.
            self.nodevec1 = nn.Parameter(left*torch.where(root>0,root,torch.ones_like(root)))
            self.nodevec2 = nn.Parameter(root[:,None]*right)
        self.encoder = Encoder(self.features,hidden_size,self.nodes,k_hop,n_rnn_layers,n_gconv_layers)
        self.decoder = Decoder(self.outputs,hidden_size,self.nodes,k_hop,n_rnn_layers,n_gconv_layers)
        self.graph_weights = nn.ParameterList(nn.Parameter(torch.eye(self.graph_dim)) for _ in range(n_gconv_layers-1) if k_hop)
        self.graph_biases = nn.ParameterList(nn.Parameter(torch.zeros(self.graph_dim)) for _ in range(n_gconv_layers-1) if k_hop)

    def learned_graphs(self):
        if not self.depth:
            return [None]*self.graph_layers
        left,right = self.nodevec1,self.nodevec2
        graphs = [F.leaky_relu(left@right)]
        for weight,bias in zip(self.graph_weights,self.graph_biases,strict=True):
            left,right = left@weight+bias,(right.T@weight+bias).T
            graphs.append(F.leaky_relu(left@right))
        return graphs

    def forecast(self,inputs,targets=None,teacher_mask=None):
        source = inputs['X']
        if source.ndim!=4 or source.shape[1:]!=(self.input_window,self.nodes,self.features):
            raise ValueError('CCRNN X must match configured [batch,history,node,feature] dimensions')
        if targets is not None and targets.shape!=(len(source),self.output_window,self.nodes,self.outputs):
            raise ValueError('CCRNN teacher targets must match [batch,horizon,node,target] dimensions')
        graphs = self.learned_graphs()
        states = self.encoder(source,graphs)
        return self.decoder(source,states,graphs,self.output_window,self.outputs,targets,teacher_mask)

    def forward(self,inputs):
        return self.forecast(inputs)

    def forward_with_teacher(self,inputs,targets,teacher_mask):
        return self.forecast(inputs,targets,teacher_mask)
