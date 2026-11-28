"""Grid OD context with column-normalized correlations and causal covariates."""
import numpy as np
import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self,inputs,hidden):
        super().__init__()
        self.hidden_dim = hidden
        self.conv = nn.Conv2d(inputs+hidden,4*hidden,3,padding=1)

    def forward(self,values,state):
        hidden,cell = state
        incoming,forget,out,candidate = self.conv(torch.cat((values,hidden),1)).chunk(4,1)
        cell = forget.sigmoid()*cell+incoming.sigmoid()*candidate.tanh()
        return out.sigmoid()*cell.tanh(),cell


class ConvLSTM(nn.Module):
    def __init__(self,inputs,hidden=32):
        super().__init__()
        self.cell_list = nn.ModuleList([ConvLSTMCell(inputs,hidden)])

    def forward(self,values):
        shape = (values.shape[0],self.cell_list[0].hidden_dim,*values.shape[-2:])
        state = (values.new_zeros(shape),values.new_zeros(shape))
        for frame in values.unbind(1):
            state = self.cell_list[0](frame,state)
        return state[0]


class CNN(nn.Module):
    def __init__(self,height,width,layers):
        super().__init__()
        self.height,self.width = height,width
        self.conv = nn.ModuleList([nn.Conv2d(1,16,3,padding=1)])
        for _ in range(1,layers):
            self.conv.extend([nn.ReLU(),nn.Conv2d(16,16,3,padding=1)])
        self.embed = nn.Sequential(nn.Conv2d(height*width*16,32,1),nn.ReLU())

    def forward(self,values):
        residual = values.reshape(-1,1,self.height,self.width)
        values = self.conv[0](residual)
        for index in range(1,len(self.conv),2):
            residual = self.conv[index](values+residual)
            values = self.conv[index+1](residual)
        values = (values+residual).relu()
        return self.embed(values.reshape(-1,self.height*self.width*16,self.height,self.width))


class LSC(nn.Module):
    def __init__(self,height,width,layers):
        super().__init__()
        self.O_CNN,self.D_CNN = CNN(height,width,layers),CNN(height,width,layers)
        self.embeder = nn.Sequential(nn.Conv2d(64,32,1),nn.ReLU())

    def forward(self,values):
        origin = self.O_CNN(values)
        destination = self.D_CNN(values.permute(0,1,4,5,2,3))
        return self.embeder(torch.cat((origin,destination),1))


class MLP(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.fc_1 = nn.Sequential(nn.Linear(features,32),nn.ReLU())
        self.fc_2 = nn.Sequential(nn.Linear(32,16),nn.ReLU())
        self.fc_3 = nn.Sequential(nn.Linear(16,8),nn.ReLU())

    def forward(self,values):
        return self.fc_3(self.fc_2(self.fc_1(values)))


class TEC(nn.Module):
    def __init__(self,inputs,channels):
        super().__init__()
        self.ConvLSTM = ConvLSTM(inputs)
        self.conv = nn.Sequential(nn.Conv2d(32,channels,1),nn.ReLU())

    def forward(self,values):
        return self.conv(self.ConvLSTM(values))


class GCC(nn.Module):
    def forward(self,values):
        local = values.flatten(2)
        similarity = local.transpose(1,2)@local
        # Each output column is a weighted sum of regional source features.
        # Normalize across source regions, never across the batch dimension.
        logits = similarity.float() if similarity.dtype in (torch.float16,torch.bfloat16) else similarity
        weights = logits.softmax(-2).to(local.dtype)
        global_features = (local@weights).reshape_as(values)
        return torch.cat((global_features,values),1)


class CSTN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,n_layers=3,c_lt=75):
        super().__init__()
        for name,value in (('input_window',input_window),('output_window',output_window),('n_layers',n_layers),('c_lt',c_lt)):
            if type(value) is not int or value<1:
                raise ValueError(f'CSTN {name} must be a positive integer')
        layout = metadata.input_layout
        if layout.get('kind')!='od_window' or metadata.feature_dim!=1 or metadata.output_dim!=1:
            raise ValueError('CSTN requires od_window data with exactly one flow feature and target')
        shape = layout.get('grid_shape')
        if not isinstance(shape,(tuple,list)) or len(shape)!=2 or any(type(value) is not int or value<1 for value in shape):
            raise ValueError('CSTN requires a complete rectangular grid_shape')
        self.height,self.width = shape
        self.num_nodes = self.height*self.width
        positions = np.asarray(layout.get('grid_positions'),dtype=np.float64)
        if (metadata.num_nodes!=self.num_nodes or positions.shape!=(self.num_nodes,2) or
                not np.isfinite(positions).all() or np.any(positions<0) or np.any(positions!=np.floor(positions)) or
                np.any(positions[:,0]>=self.height) or np.any(positions[:,1]>=self.width)):
            raise ValueError('CSTN grid_positions must map every node into the declared rectangle')
        node_to_grid = (positions[:,0]*self.width+positions[:,1]).astype(np.int64)
        if len(np.unique(node_to_grid))!=self.num_nodes:
            raise ValueError('CSTN grid_positions must contain unique cells')
        self.register_buffer('node_to_grid',torch.tensor(node_to_grid))
        self.register_buffer('grid_to_node',torch.tensor(np.argsort(node_to_grid)))
        features = layout.get('history_features',{})
        columns = tuple(features.get('columns',()))
        known = tuple(layout.get('future_feature_names',()))
        if (features.get('kind')!='global_covariates' or any(not isinstance(name,str) for name in columns+known) or
                len(set(columns))!=len(columns) or len(set(known))!=len(known) or set(known)-set(columns)):
            raise ValueError('CSTN requires explicit historical and known-future covariate names')
        self.history_dim,self.future_dim = len(columns),len(known)
        self.register_buffer('future_positions',torch.tensor([columns.index(name) for name in known],dtype=torch.long))
        self.input_window,self.output_window = input_window,output_window
        self.LSC = LSC(self.height,self.width,n_layers)
        self.MLP = MLP(self.history_dim) if self.history_dim else None
        self.TEC = TEC(40 if self.history_dim else 32,c_lt)
        self.GCC = GCC()
        self.OUT = nn.Sequential(nn.Conv2d(2*c_lt,self.num_nodes,1),nn.ReLU(),nn.Tanh())

    def forward(self,inputs):
        values = inputs['X']
        if values.ndim!=5 or values.shape[1:]!=(self.input_window,self.num_nodes,self.num_nodes,1):
            raise ValueError('CSTN X must have shape [B,input_window,N,N,1]')
        batch = values.shape[0]
        weather,future = inputs.get('history_features'),inputs.get('future_features')
        if self.history_dim and (weather is None or weather.shape!=(batch,self.input_window,self.history_dim)):
            raise ValueError('CSTN history_features must match historical covariate metadata')
        if self.future_dim and (future is None or future.shape!=(batch,self.output_window,self.future_dim)):
            raise ValueError('CSTN future_features must match declared known-future columns and horizon')
        values = values[...,0].index_select(2,self.grid_to_node).index_select(3,self.grid_to_node)
        predictions = []
        for step in range(self.output_window):
            grid = values.reshape(batch,self.input_window,self.height,self.width,self.height,self.width)
            spatial = self.LSC(grid).reshape(batch,self.input_window,32,self.height,self.width)
            if self.history_dim:
                embedded = self.MLP(weather)[...,None,None].expand(-1,-1,-1,self.height,self.width)
                spatial = torch.cat((spatial,embedded),2)
            prediction = self.OUT(self.GCC(self.TEC(spatial))).reshape(batch,1,self.num_nodes,self.num_nodes)
            predictions.append(prediction)
            if step+1<self.output_window:
                values = torch.cat((values[:,1:],prediction),1)
                if self.history_dim:
                    # Unknown future observations use persistence from the last
                    # observed frame. Only declared known columns advance.
                    next_weather = weather[:,-1:]
                    if self.future_dim:
                        indices = self.future_positions.expand(batch,1,-1)
                        next_weather = next_weather.scatter(-1,indices,future[:,step:step+1].to(weather.dtype))
                    weather = torch.cat((weather[:,1:],next_weather),1)
        result = torch.cat(predictions,1)
        return result.index_select(2,self.node_to_grid).index_select(3,self.node_to_grid)[...,None]
