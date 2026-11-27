"""Causal temporal attention and node identity over explicit full-node windows."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.traffic_flow_prediction.SimST import CT_MSA


def neighbor_projection(adjacency,count):
    """Outgoing/incoming top neighbors, then their unweighted means."""
    nodes = len(adjacency)
    result = np.zeros((2*count+2,nodes,nodes),dtype=np.float32)
    for node in range(nodes):
        directions = []
        for graph in (adjacency,adjacency.T):
            neighbors = np.flatnonzero(graph[node])
            neighbors = neighbors[neighbors!=node]
            ranked = neighbors[np.argsort(graph[node,neighbors],kind='stable')[-count:]] if count else []
            ranked = list(map(int,ranked))
            ranked += [node]*max(0,count-len(ranked))
            directions.append((ranked,neighbors.tolist() or [node]))
        for rank in range(count):
            for direction in range(2):
                result[rank*2+direction,node,directions[direction][0][count-rank-1]] = 1.
        for direction in range(2):
            neighbors = directions[direction][1]
            result[2*count+direction,node,neighbors] = 1./len(neighbors)
    return result


class SimST(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,node_dim=20,init_dim=64,end_dim=512,
                 layer=2,dropout=.1,in_neighbor_num=0):
        super().__init__()
        if min(init_dim,end_dim,layer)<1 or init_dim%2 or node_dim<0 or in_neighbor_num<0 or not 0<=dropout<1:
            raise ValueError('SimST requires positive widths/layers, even init_dim and nonnegative node/neighbor settings')
        if metadata.input_layout.get('kind')!='window':
            raise ValueError('SimST requires contiguous input windows')
        self.output_window,self.output_dim = output_window,metadata.output_dim
        self.node_dim = node_dim
        self.day_indices = tuple(metadata.input_layout.get('calendar',{}).get('day_of_week',()))
        projection = neighbor_projection(metadata.adjacency,in_neighbor_num)
        self.register_buffer('neighbor_projection',torch.from_numpy(projection))
        features = metadata.feature_dim-(6 if self.day_indices else 0)+len(projection)
        self.start_conv = nn.Conv2d(features,init_dim,1)
        self.encoder = CT_MSA(init_dim,layer,2,input_window,init_dim*2,input_window,dropout)
        if node_dim:
            self.node_embed = nn.Parameter(torch.randn(metadata.num_nodes,node_dim))
            self.node_linear = nn.Linear(node_dim,init_dim)
        self.end_conv_1 = nn.Conv2d(init_dim*(2 if node_dim else 1),end_dim,1)
        self.end_conv_2 = nn.Conv2d(end_dim,output_window*metadata.output_dim,1)

    def packed_inputs(self,source):
        # The old architecture embeds weekday as weekday/7 rather than one-hot.
        packed = source
        if self.day_indices:
            day = source[...,list(self.day_indices)].argmax(-1,keepdim=True).to(source.dtype)/7
            packed = torch.cat((source[...,:self.day_indices[0]],day,source[...,self.day_indices[-1]+1:]),dim=-1)
        neighbors = torch.einsum('btm,cnm->btnc',source[...,0],self.neighbor_projection)
        return torch.cat((packed,neighbors),dim=-1)

    def forward(self,batch):
        source = self.packed_inputs(batch['X']).permute(0,3,2,1)
        encoded = self.encoder(self.start_conv(source))[...,-1:]
        if self.node_dim:
            identity = self.node_linear(self.node_embed).T[None,:,:,None].expand(source.shape[0],-1,-1,-1)
            encoded = torch.cat((encoded,identity),dim=1)
        output = self.end_conv_2(F.relu(self.end_conv_1(encoded)))
        return output[...,0].reshape(source.shape[0],self.output_window,self.output_dim,source.shape[2]).transpose(2,3)
