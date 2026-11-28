"""Synchronous three-frame graph convolutions with optional learned edge masks."""
import numpy as np
import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, length, nodes, channels, temporal, spatial):
        super().__init__()
        self.register_parameter('temporal_emb', nn.Parameter(torch.empty(1,length,1,channels)) if temporal else None)
        self.register_parameter('spatial_emb', nn.Parameter(torch.empty(1,1,nodes,channels)) if spatial else None)
        for value in self.parameters():
            nn.init.xavier_uniform_(value)

    def forward(self, value):
        if self.temporal_emb is not None:
            value = value+self.temporal_emb
        if self.spatial_emb is not None:
            value = value+self.spatial_emb
        return value


class GcnOperation(nn.Module):
    def __init__(self, inputs, outputs, activation):
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(inputs, outputs*(2 if activation=='GLU' else 1))

    def forward(self, value, adjacency):
        value = self.layer(adjacency@value)
        if self.activation=='GLU':
            left,right = value.chunk(2,-1)
            return left*right.sigmoid()
        return value.relu()


class Stsgcm(nn.Module):
    def __init__(self, inputs, filters, nodes, activation):
        super().__init__()
        self.nodes = nodes
        self.layers = nn.ModuleList(GcnOperation(before,after,activation)
            for before,after in zip([inputs]+filters[:-1],filters))

    def forward(self, value, adjacency):
        centers = []
        for layer in self.layers:
            value = layer(value,adjacency)
            centers.append(value[:,self.nodes:2*self.nodes])
        return torch.stack(centers).max(0).values


class STSGCL(nn.Module):
    def __init__(self, length, nodes, inputs, filters, module_type, activation, temporal, spatial):
        super().__init__()
        self.module_type = module_type
        self.position_embedding = PositionEmbedding(length,nodes,inputs,temporal,spatial)
        if module_type=='individual':
            self.gcms = nn.ModuleList(Stsgcm(inputs,filters,nodes,activation) for _ in range(length-2))
        else:
            self.gcm = Stsgcm(inputs,filters,nodes,activation)

    def forward(self, value, adjacency):
        value = self.position_embedding(value)
        batch,length,nodes,channels = value.shape
        if self.module_type=='individual':
            return torch.stack([module(value[:,index:index+3].reshape(batch,3*nodes,channels),adjacency)
                for index,module in enumerate(self.gcms)],1)
        # Time-major packing preserves the original sharing ablation's window order.
        windows = value.unfold(1,3,1).permute(1,0,4,2,3).reshape((length-2)*batch,3*nodes,channels)
        return self.gcm(windows,adjacency).reshape(length-2,batch,nodes,-1).transpose(0,1)


class OutputLayer(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.hidden_layer = nn.Linear(inputs,hidden)
        self.output_layer = nn.Linear(hidden,outputs)

    def forward(self, value):
        value = value.transpose(1,2).flatten(2)
        return self.output_layer(self.hidden_layer(value).relu()).unsqueeze(1)


def local_adjacency(graph):
    """Preserve directed spatial weights, adjacent-frame identity edges and self loops."""
    nodes = graph.shape[0]
    result = np.kron(np.eye(3,dtype=np.float32),graph)
    for index in range(2):
        before = np.arange(nodes)+index*nodes
        after = before+nodes
        result[before,after] = result[after,before] = 1
    np.fill_diagonal(result,1)
    return result


class STSGCN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, filters=None, module_type='individual',
                 act_type='GLU', temporal_emb=True, spatial_emb=True, use_mask=False,
                 first_layer_embedding_size=64, output_hidden_size=128):
        super().__init__()
        filters = [[64,64,64] for _ in range(4)] if filters is None else filters
        if (not isinstance(filters,(list,tuple)) or not filters or any(not isinstance(block,(list,tuple))
                or not block or any(type(width) is not int or width<1 for width in block)
                or len(set(block))!=1 for block in filters)):
            raise ValueError('STSGCN filters require nonempty blocks of equal positive widths within each block')
        if type(input_window) is not int or input_window<2*len(filters)+1:
            raise ValueError('STSGCN input_window must exceed twice the number of convolution blocks')
        if type(output_window) is not int or output_window<1:
            raise ValueError('STSGCN output_window must be positive')
        if module_type not in ('individual','sharing') or act_type not in ('GLU','relu'):
            raise ValueError('STSGCN requires module_type=individual/sharing and act_type=GLU/relu')
        if any(type(value) is not bool for value in (temporal_emb,spatial_emb,use_mask)):
            raise ValueError('STSGCN embedding and mask switches must be booleans')
        if type(first_layer_embedding_size) is not int or first_layer_embedding_size<0:
            raise ValueError('STSGCN first_layer_embedding_size must be a nonnegative integer')
        if type(output_hidden_size) is not int or output_hidden_size<1:
            raise ValueError('STSGCN output_hidden_size must be positive')
        if metadata.input_layout.get('kind','window')!='window':
            raise ValueError('STSGCN requires continuous window sampling')
        self.input_window,self.num_nodes,self.feature_dim = input_window,metadata.num_nodes,metadata.feature_dim
        graph = np.array(metadata.adjacency,dtype=np.float32,copy=True)
        if graph.shape!=(self.num_nodes,self.num_nodes) or not np.isfinite(graph).all():
            raise ValueError('STSGCN requires a finite square adjacency matching num_nodes')
        self.register_buffer('adjacency',torch.from_numpy(local_adjacency(graph)))
        self.register_parameter('mask',nn.Parameter((self.adjacency!=0).float()) if use_mask else None)
        channels = first_layer_embedding_size or self.feature_dim
        self.first_layer_embedding = nn.Linear(self.feature_dim,channels) if first_layer_embedding_size else None
        self.stsgcl_layers = nn.ModuleList()
        length = input_window
        for block in filters:
            block = list(block)
            self.stsgcl_layers.append(STSGCL(length,self.num_nodes,channels,block,module_type,act_type,
                temporal_emb,spatial_emb))
            length,channels = length-2,block[-1]
        self.outputs = nn.ModuleList(OutputLayer(length*channels,output_hidden_size,metadata.output_dim)
            for _ in range(output_window))

    def forward(self, inputs):
        value = inputs['X']
        if value.ndim!=4 or value.shape[1:]!=(self.input_window,self.num_nodes,self.feature_dim):
            raise ValueError('STSGCN X must have configured [batch,history,node,feature] dimensions')
        adjacency = self.adjacency if self.mask is None else self.adjacency*self.mask
        if self.first_layer_embedding is not None:
            value = self.first_layer_embedding(value).relu()
        for layer in self.stsgcl_layers:
            value = layer(value,adjacency)
        return torch.cat([head(value) for head in self.outputs],1)
