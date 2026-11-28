"""OD graph embeddings with sample-local periodic memory and label-free rollout."""
import numpy as np
import torch
from torch import nn


def geographic_graph(distance):
    # Row scaling preserves d^2/sum(d^2) without overflowing large distances.
    scale = distance.amax(-1,keepdim=True)
    squared = (distance/torch.where(scale>0,scale,1.)).square()
    total = squared.sum(-1,keepdim=True)
    weights = squared/torch.where(total>0,total,1.)
    identity = torch.eye(distance.shape[-1],device=distance.device,dtype=distance.dtype)
    return weights*(1-identity)+identity


def semantic_graph(mask,dtype):
    # Topology uses raw positive flows, never normalized signs or future y.
    adjacency = (mask|mask.transpose(-1,-2)).to(dtype)
    degree = adjacency.sum(-1,keepdim=True)
    weights = (1/(adjacency@degree+1e-3))@degree.transpose(-1,-2)
    identity = torch.eye(mask.shape[-1],device=mask.device,dtype=dtype)
    return weights*(1-identity)+identity


class GraphConvolution(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inputs,outputs))

    def forward(self,values,graph):
        return (graph@values)@self.weight


class GCN(nn.Module):
    def __init__(self,nodes,embedding):
        super().__init__()
        self.gcn = nn.ModuleList([GraphConvolution(nodes,embedding),GraphConvolution(embedding,embedding)])

    def forward(self,values,graph):
        for layer in self.gcn:
            values = layer(values,graph)
        return values


class SLSTM(nn.Module):
    def __init__(self,width,period):
        super().__init__()
        self.p_interval = period
        for name,activation in (('f_gate',nn.Softmax(-1)),('i_gate',nn.Softmax(-1)),
                ('o_gate',nn.Softmax(-1)),('g_gate',nn.Tanh())):
            setattr(self,name,nn.Sequential(nn.Linear(2*width,width),activation))

    def forward(self,values):
        phases = min(self.p_interval,len(values))
        hidden = [values.new_zeros(values.shape[1:]) for _ in range(phases)]
        cells = [values.new_zeros(values.shape[1:]) for _ in range(phases)]
        for time,value in enumerate(values):
            index = time%self.p_interval
            source = torch.cat((value,hidden[index]),-1)
            # Keep GEML's stored tanh-cell convention and feature softmax gates.
            # Independent states replace overwritten cells/in-place hidden writes.
            cells[index] = torch.tanh(self.f_gate(source)*cells[index]+self.i_gate(source)*self.g_gate(source))
            hidden[index] = self.o_gate(source)*cells[index]
        return hidden[(len(values)-1)%self.p_interval]


class MultiLearning(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.transition = nn.Parameter(torch.randn(width,width))
        self.project_in = nn.Parameter(torch.randn(width,1))
        self.project_out = nn.Parameter(torch.randn(width,1))

    def forecast(self,values):
        return ((values@self.transition)@values.transpose(-1,-2))[:,None,...,None]

    def forward(self,values):
        return (self.forecast(values),(values@self.project_in)[:,None,...,None],
            (values@self.project_out)[:,None,...,None])


class GEML(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,embed_dim=32,p_interval=1):
        super().__init__()
        for name,value in (('input_window',input_window),('output_window',output_window),
                ('embed_dim',embed_dim),('p_interval',p_interval)):
            if type(value) is not int or value<1:
                raise ValueError(f'GEML {name} must be a positive integer')
        if metadata.input_layout.get('kind')!='od_window' or metadata.feature_dim!=1 or metadata.output_dim!=1:
            raise ValueError('GEML requires od_window data with exactly one flow feature and target')
        self.num_nodes,self.input_window,self.output_window = metadata.num_nodes,input_window,output_window
        self.embed_dim = embed_dim
        if metadata.adjacency is None:
            raise ValueError('GEML requires an explicit geographic adjacency matrix')
        distance = torch.tensor(np.array(metadata.adjacency,copy=True),dtype=torch.float64)
        if distance.shape!=(self.num_nodes,self.num_nodes) or not torch.isfinite(distance).all() or (distance<0).any():
            raise ValueError('GEML geographic distances must be finite nonnegative NxN values')
        self.register_buffer('geo_adj',geographic_graph(distance).float())
        self.GCN_ge,self.GCN_se = GCN(self.num_nodes,embed_dim),GCN(self.num_nodes,embed_dim)
        self.LSTM = SLSTM(2*embed_dim,p_interval)
        self.mutiLearning = MultiLearning(2*embed_dim)

    def _rollout(self,inputs,auxiliary):
        values,mask = inputs['X'],inputs['history_mask']
        if values.ndim!=5 or values.shape[1:]!=(self.input_window,self.num_nodes,self.num_nodes,1):
            raise ValueError('GEML X must have shape [B,input_window,N,N,1]')
        if mask.shape!=values.shape or mask.dtype!=torch.bool:
            raise ValueError('GEML history_mask must be a boolean mask matching X')
        # Keep the initial observed-history support for the entire rollout;
        # synthetic future predictions never introduce graph edges.
        graph = semantic_graph(mask[...,0],values.dtype)
        predictions,origins,destinations = [],[],[]
        for _ in range(self.output_window):
            source = values[...,0]
            embedding = torch.cat((self.GCN_ge(source,self.geo_adj),self.GCN_se(source,graph)),-1)
            temporal = embedding.permute(1,0,2,3).reshape(self.input_window,-1,2*self.embed_dim)
            state = self.LSTM(temporal).reshape(values.shape[0],self.num_nodes,2*self.embed_dim)
            if auxiliary:
                prediction,origin,destination = self.mutiLearning(state)
                origins.append(origin);destinations.append(destination)
            else:
                prediction = self.mutiLearning.forecast(state)
            predictions.append(prediction)
            values = torch.cat((values[:,1:],prediction),1)
        prediction = torch.cat(predictions,1)
        return (prediction,torch.cat(origins,1),torch.cat(destinations,1)) if auxiliary else prediction

    def forward(self,inputs):
        return self._rollout(inputs,False)

    def forward_with_aux(self,inputs):
        return self._rollout(inputs,True)
