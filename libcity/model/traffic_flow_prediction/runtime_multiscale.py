"""Multiscale graph forecasts use declared CPT segments and registered graph bases."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.traffic_flow_prediction.ASTGCN import SpatialAttentionLayer, TemporalAttentionLayer


def spectral_basis(adjacency, order):
    graph = np.array(adjacency,dtype=np.float64,copy=True)
    if graph.ndim!=2 or graph.shape[0]!=graph.shape[1] or not np.isfinite(graph).all() or np.any(graph<0):
        raise ValueError('Multiscale graph adjacency must be finite, square and nonnegative')
    degree = graph.sum(axis=1)
    laplacian = np.diag(degree)-graph
    denominator = np.sqrt(degree[:,None]*degree[None,:])
    np.divide(laplacian,denominator,out=laplacian,where=denominator>0)
    maximum = float(np.linalg.eigvals(laplacian).real.max())
    scaled = 2*laplacian/maximum-np.eye(len(graph)) if maximum>1e-12 else -np.eye(len(graph))
    terms = [np.eye(len(graph))]
    if order>1:
        terms.append(scaled)
    for _ in range(2,order):
        terms.append(2*scaled@terms[-1]-terms[-2])
    return torch.tensor(np.stack(terms),dtype=torch.float32)


class GraphConvolution(nn.Module):
    def __init__(self,basis,inputs,outputs):
        super().__init__()
        self.register_buffer('basis',basis)
        self.Theta = nn.ParameterList(nn.Parameter(torch.empty(inputs,outputs)) for _ in range(len(basis)))
        self.outputs = outputs

    def forward(self,source,attention=None):
        result = source.new_zeros(source.shape[0],source.shape[1],self.outputs,source.shape[-1])
        for graph,weight in zip(self.basis,self.Theta):
            if attention is not None:
                matrix = (graph*attention).transpose(1,2)
                propagated = torch.einsum('bnm,bmit->bnit',matrix,source)
            else:
                propagated = torch.einsum('nm,bmit->bnit',graph.T,source)
            result = result+torch.einsum('bnit,io->bnot',propagated,weight)
        return F.relu(result)


class Block(nn.Module):
    def __init__(self,basis,inputs,graph_filters,time_filters,stride,nodes,steps,attention):
        super().__init__()
        self.attention = attention
        if attention:
            self.TAt = TemporalAttentionLayer('cpu',inputs,nodes,steps)
            self.SAt = SpatialAttentionLayer('cpu',inputs,nodes,steps)
            self.cheb_conv_SAt = GraphConvolution(basis,inputs,graph_filters)
        else:
            self.ChebConv = GraphConvolution(basis,inputs,graph_filters)
        self.time_conv = nn.Conv2d(graph_filters,time_filters,(1,3),stride=(1,stride),padding=(0,1))
        self.residual_conv = nn.Conv2d(inputs,time_filters,1,stride=(1,stride))
        self.ln = nn.LayerNorm(time_filters)

    def forward(self,source):
        if self.attention:
            count,nodes,features,steps = source.shape
            temporal = self.TAt(source)
            attended = (source.reshape(count,-1,steps)@temporal).reshape(count,nodes,features,steps)
            graph = self.cheb_conv_SAt(source,self.SAt(attended))
        else:
            graph = self.ChebConv(source)
        hidden = self.time_conv(graph.permute(0,2,1,3))+self.residual_conv(source.permute(0,2,1,3))
        return self.ln(F.relu(hidden).permute(0,3,2,1)).permute(0,2,3,1)


class Fusion(nn.Module):
    def __init__(self,horizon,nodes,outputs):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(1,horizon,nodes,outputs))

    def forward(self,source):
        return source*self.weights


class Branch(nn.Module):
    def __init__(self,basis,blocks,inputs,graph_filters,time_filters,count,horizon,outputs,nodes,attention):
        super().__init__()
        self.BlockList = nn.ModuleList([Block(basis,inputs,graph_filters,time_filters,count,nodes,count*horizon,attention)])
        self.BlockList.extend(Block(basis,time_filters,graph_filters,time_filters,1,nodes,horizon,attention)
                              for _ in range(blocks-1))
        self.final_conv = nn.Conv2d(horizon,horizon,(1,time_filters-outputs+1))
        self.fusionlayer = Fusion(horizon,nodes,outputs)

    def forward(self,source):
        hidden = source.permute(0,2,3,1)
        for block in self.BlockList:
            hidden = block(hidden)
        return self.fusionlayer(self.final_conv(hidden.permute(0,3,1,2)))


class MultiscaleGraph(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,nb_block=2,K=3,nb_chev_filter=64,
                 nb_time_filter=64,attention):
        super().__init__()
        if metadata.input_layout.get('kind')!='cpt':
            raise ValueError('ASTGCN/MSTGCN require data.sampling.name=cpt')
        if min(nb_block,K,nb_chev_filter,nb_time_filter)<1 or nb_time_filter<metadata.output_dim:
            raise ValueError('Multiscale graph dimensions must be positive and nb_time_filter >= output_dim')
        sampling = metadata.input_layout['sampling']
        basis = spectral_basis(metadata.adjacency,K)
        family = 'ASTGCN' if attention else 'MSTGCN'
        self.segments = []
        start = 0
        # The data contract stores trend/period/closeness, while the legacy
        # models expected closeness/period/trend. Slice by the declared layout.
        for kind,prefix in (('trend','weeks'),('period','days'),('closeness','hours')):
            count = sampling.get(kind,0)
            if count:
                attribute = f'{prefix}_{family}_submodule'
                setattr(self,attribute,Branch(basis,nb_block,metadata.feature_dim,nb_chev_filter,nb_time_filter,
                    count,output_window,metadata.output_dim,metadata.num_nodes,attention))
                length = count*output_window
                self.segments.append((attribute,start,start+length))
                start += length
        if start!=metadata.input_layout['length'] or not self.segments:
            raise ValueError('CPT segment lengths disagree with the data layout')
        for parameter in self.parameters():
            if parameter.ndim>1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def forward(self,batch):
        source = batch['X']
        outputs = [getattr(self,name)(source[:,start:end]) for name,start,end in self.segments]
        return torch.stack(outputs).sum(dim=0)


class ASTGCN(MultiscaleGraph):
    def __init__(self,**parameters):
        super().__init__(attention=True,**parameters)


class MSTGCN(MultiscaleGraph):
    def __init__(self,**parameters):
        super().__init__(attention=False,**parameters)
