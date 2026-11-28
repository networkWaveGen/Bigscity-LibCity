"""GSNet's spatial grid and semantic graphs with explicit forecast-time inputs."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def temporal_attention(values, target, content, time, bias):
    # values: B,N,T,D. Never squeeze a batch, node or time dimension.
    logits = content(values).squeeze(-1)+bias
    if time is not None:
        logits = logits+time(target)[:,None,:]
    return (values*F.relu(logits).softmax(-1).unsqueeze(-1)).sum(-2)


class GCNLayer(nn.Module):
    def __init__(self, features, filters):
        super().__init__()
        self.gcn_layer = nn.Sequential(nn.Linear(features,filters),nn.ReLU())

    def forward(self, values, adjacency):
        return self.gcn_layer(adjacency@values)


class STGeoModule(nn.Module):
    def __init__(self, channels, layers, steps, hidden, future):
        super().__init__()
        self.grid_conv = nn.Sequential(nn.Conv2d(channels,64,3,padding=1),nn.ReLU(),
            nn.Conv2d(64,channels,3,padding=1),nn.ReLU())
        self.grid_gru = nn.GRU(channels,hidden,layers,batch_first=True)
        self.grid_att_fc1 = nn.Linear(hidden,1)
        self.grid_att_fc2 = nn.Linear(future,steps) if future else None
        self.grid_att_bias = nn.Parameter(torch.zeros(1))

    def forward(self, grid, target):
        batch,steps,channels,rows,columns = grid.shape
        values = self.grid_conv(grid.reshape(batch*steps,channels,rows,columns))
        values = values.reshape(batch,steps,channels,rows*columns).permute(0,3,1,2)
        values,_ = self.grid_gru(values.reshape(batch*rows*columns,steps,channels))
        values = values.reshape(batch,rows*columns,steps,-1)
        output = temporal_attention(values,target,self.grid_att_fc1,self.grid_att_fc2,self.grid_att_bias)
        return output.transpose(1,2).reshape(batch,-1,rows,columns)


class STSemModule(nn.Module):
    def __init__(self, features, filters, steps, layers, hidden, future, shape, poi):
        super().__init__()
        self.shape = shape
        widths = (features,*filters)
        for name in ('road','risk','poi'):
            modules = [GCNLayer(left,right) for left,right in zip(widths,widths[1:])] if name!='poi' or poi else []
            setattr(self,name+'_gcn',nn.ModuleList(modules))
        self.graph_gru = nn.GRU(filters[-1],hidden,layers,batch_first=True)
        self.graph_att_fc1 = nn.Linear(hidden,1)
        self.graph_att_fc2 = nn.Linear(future,steps) if future else None
        self.graph_att_bias = nn.Parameter(torch.zeros(1))

    def forward(self, graph, target, road, risk, poi, mapping):
        batch,steps,nodes,features = graph.shape
        combined = None
        for name,adjacency in (('road',road),('risk',risk),('poi',poi)):
            if adjacency is None:
                continue
            values = graph
            for layer in getattr(self,name+'_gcn'):
                values = layer(values,adjacency)
            combined = values if combined is None else combined+values
        values = combined.transpose(1,2).reshape(batch*nodes,steps,-1)
        values,_ = self.graph_gru(values)
        values = values.reshape(batch,nodes,steps,-1)
        values = temporal_attention(values,target,self.graph_att_fc1,self.graph_att_fc2,self.graph_att_bias)
        return (mapping@values).transpose(1,2).reshape(batch,-1,*self.shape)


class RiskNetwork(nn.Module):
    def __init__(self, channels, graph_features, layers, steps, horizon, hidden, future, filters, shape, poi):
        super().__init__()
        self.horizon,self.shape = horizon,shape
        self.st_geo_module = STGeoModule(channels,layers,steps,hidden,future)
        self.st_sem_module = STSemModule(graph_features,filters,steps,layers,hidden,future,shape,poi)
        self.grid_weight = nn.Conv2d(hidden,16,1)
        self.graph_weight = nn.Conv2d(hidden,16,1)
        self.output_layer = nn.Linear(16*shape[0]*shape[1],horizon*shape[0]*shape[1])

    def forward(self, grid, target, graph, road, risk, poi, mapping):
        geographical = self.grid_weight(self.st_geo_module(grid,target))
        semantic = self.graph_weight(self.st_sem_module(graph,target,road,risk,poi,mapping))
        return self.output_layer((geographical+semantic).flatten(1)).reshape(len(grid),self.horizon,-1,1)


class GSNet(nn.Module):
    def __init__(self, metadata, input_window, output_window, num_of_gru_layers=5,
                 gru_hidden_size=256, gcn_nums_filters=(64,64)):
        super().__init__()
        for name,value in (('input_window',input_window),('output_window',output_window),
                ('num_of_gru_layers',num_of_gru_layers),('gru_hidden_size',gru_hidden_size)):
            if type(value) is not int or value<1:
                raise ValueError('GSNet '+name+' must be a positive integer')
        if (not isinstance(gcn_nums_filters,(tuple,list)) or not gcn_nums_filters or
                any(type(value) is not int or value<1 for value in gcn_nums_filters)):
            raise ValueError('GSNet gcn_nums_filters must contain positive integer widths')
        layout = metadata.input_layout
        if layout.get('kind')!='risk_cpt' or layout.get('spatial_identity')!='row_major_grid_cells' or metadata.output_dim!=1:
            raise ValueError('GSNet requires a risk_cpt full grid with one leading risk target')
        self.shape = tuple(layout['grid_shape'])
        self.nodes,self.features = metadata.num_nodes,metadata.feature_dim
        self.steps,self.horizon = layout['length'],output_window
        self.external,self.future = layout['historical_external_dim'],layout['future_external_dim']
        if len(self.shape)!=2 or np.prod(self.shape)!=self.nodes or self.steps<1:
            raise ValueError('GSNet grid shape and history must match the data layout')
        indices = tuple(layout['graph_input_indices'])
        if not indices or len(set(indices))!=len(indices) or any(type(index) is not int or not 0<=index<self.features for index in indices):
            raise ValueError('GSNet requires distinct graph feature indices in observed inputs')
        self.register_buffer('graph_input_indices',torch.tensor(indices,dtype=torch.long))
        static = metadata.static_data
        graph_nodes = layout['graph_nodes']
        for name,shape in (('grid_node_map',(self.nodes,graph_nodes)),('road_adj',(graph_nodes,graph_nodes)),
                ('risk_adj',(graph_nodes,graph_nodes)),('poi_adj',(graph_nodes,graph_nodes))):
            raw = static.get(name)
            if raw is None and name=='poi_adj':
                self.register_buffer(name,None)
                continue
            if raw is None or np.shape(raw)!=shape or not np.isfinite(raw).all():
                raise ValueError('GSNet requires finite, correctly shaped '+name)
            self.register_buffer(name,torch.tensor(np.array(raw,copy=True),dtype=torch.float32))
        self.gsnet = RiskNetwork(self.features+self.external,len(indices),num_of_gru_layers,self.steps,
            output_window,gru_hidden_size,self.future,tuple(gcn_nums_filters),self.shape,self.poi_adj is not None)

    def forward(self, batch):
        observed = batch['X']
        if observed.ndim!=4 or observed.shape[1:]!=(self.steps,self.nodes,self.features):
            raise ValueError('GSNet X must match [B, effective history, full grid cells, observed features]')
        batch_size = len(observed)
        values = observed
        if self.external:
            historical = batch['history_features']
            if historical.shape!=(batch_size,self.steps,self.external):
                raise ValueError('GSNet history_features must match [B, effective history, external features]')
            values = torch.cat((values,historical[:,:,None,:].expand(-1,-1,self.nodes,-1)),dim=-1)
        target = None
        if self.future:
            future = batch['future_features']
            if future.shape!=(batch_size,self.horizon,self.future):
                raise ValueError('GSNet future_features must match [B, forecast horizon, known features]')
            # The original fusion head uses one forecast-time context for all
            # outputs. Use the first real forecast timestamp, never a past frame.
            target = future[:,0]
        grid = values.transpose(2,3).reshape(batch_size,self.steps,-1,*self.shape)
        selected = observed.index_select(-1,self.graph_input_indices).transpose(2,3)
        graph = (selected@self.grid_node_map).transpose(2,3)
        return self.gsnet(grid,target,graph,self.road_adj,self.risk_adj,self.poi_adj,self.grid_node_map)
