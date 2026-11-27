"""Portable diffusion GRU cells; curriculum targets are passed only by a train step."""
import math
import numpy as np
import torch
from torch import nn

from libcity.model.traffic_speed_prediction.DCRNN import calculate_random_walk_matrix, calculate_scaled_laplacian


def supports(adjacency, filter_type):
    if filter_type == 'dual_random_walk':
        matrices = [calculate_random_walk_matrix(adjacency).T, calculate_random_walk_matrix(adjacency.T).T]
    elif filter_type == 'random_walk':
        matrices = [calculate_random_walk_matrix(adjacency).T]
    elif filter_type == 'laplacian':
        matrices = [calculate_scaled_laplacian(adjacency, lambda_max=None)]
    else:
        raise ValueError('Unknown DCRNN filter_type')
    result = []
    for matrix in matrices:
        matrix = matrix.tocoo()
        indices = np.column_stack((matrix.row, matrix.col))
        indices = indices[np.lexsort((indices[:,0], indices[:,1]))]
        result.append(torch.sparse_coo_tensor(indices.T, matrix.data, matrix.shape,check_invariants=True).coalesce().float())
    return result


class DiffusionLinear(nn.Module):
    def __init__(self, nodes, depth, graphs, input_dim, hidden, output_dim, bias=0.):
        super().__init__()
        self.nodes, self.depth = nodes, depth
        self.names = []
        for index, graph in enumerate(graphs):
            name = f'support_{index}'
            self.register_buffer(name, graph)
            self.names.append(name)
        self.matrices = len(graphs)*depth+1
        self.weight = nn.Parameter(torch.empty((input_dim+hidden)*self.matrices, output_dim))
        self.biases = nn.Parameter(torch.empty(output_dim))
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.biases, bias)

    def forward(self, inputs, hidden):
        count = inputs.shape[0]
        value = torch.cat((inputs.reshape(count,self.nodes,-1), hidden.reshape(count,self.nodes,-1)), dim=2)
        size = value.shape[2]
        x0 = value.permute(1,2,0).reshape(self.nodes,size*count)
        terms = [x0]
        if self.depth:
            for name in self.names:
                graph = getattr(self,name)
                x1 = torch.sparse.mm(graph,x0)
                terms.append(x1)
                for _ in range(2,self.depth+1):
                    x2 = 2*torch.sparse.mm(graph,x1)-x0
                    terms.append(x2)
                    x1,x0 = x2,x1
        value = torch.stack(terms).reshape(self.matrices,self.nodes,size,count).permute(3,1,2,0)
        value = value.reshape(count*self.nodes,size*self.matrices)@self.weight+self.biases
        return value.reshape(count,-1)


class DiffusionCell(nn.Module):
    def __init__(self, nodes, depth, graphs, input_dim, units):
        super().__init__()
        self.nodes,self.units = nodes,units
        self._fn = DiffusionLinear(nodes,depth,graphs,input_dim,units,2*units,1.)
        self._gconv = DiffusionLinear(nodes,depth,graphs,input_dim,units,units)

    def forward(self, inputs, hidden):
        reset, update = torch.sigmoid(self._fn(inputs,hidden)).reshape(-1,self.nodes,2*self.units).chunk(2,-1)
        reset,update = reset.reshape(inputs.shape[0],-1),update.reshape(inputs.shape[0],-1)
        candidate = torch.tanh(self._gconv(inputs,reset*hidden))
        return update*hidden+(1-update)*candidate


class DiffusionStack(nn.Module):
    def __init__(self, nodes, depth, graphs, input_dim, units, layers, output_dim=None):
        super().__init__()
        self.nodes,self.units = nodes,units
        if output_dim is not None:
            self.projection_layer = nn.Linear(units,output_dim)
        self.dcgru_layers = nn.ModuleList([DiffusionCell(nodes,depth,graphs,input_dim if index==0 else units,units)
                                          for index in range(layers)])

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = inputs.new_zeros(len(self.dcgru_layers),inputs.shape[0],self.nodes*self.units)
        states = []
        for index, cell in enumerate(self.dcgru_layers):
            inputs = cell(inputs,hidden[index])
            states.append(inputs)
        if hasattr(self,'projection_layer'):
            inputs = self.projection_layer(inputs.reshape(-1,self.units)).reshape(hidden.shape[1],-1)
        return inputs,torch.stack(states)


class DCRNN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, rnn_units=64, num_rnn_layers=2,
                 max_diffusion_step=2, filter_type='dual_random_walk', use_curriculum_learning=True, cl_decay_steps=1000):
        super().__init__()
        if rnn_units<1 or num_rnn_layers<1 or max_diffusion_step<0 or cl_decay_steps<1:
            raise ValueError('DCRNN requires positive units/layers/decay and nonnegative diffusion depth')
        self.num_nodes,self.output_dim = metadata.num_nodes,metadata.output_dim
        self.input_window,self.output_window = input_window,output_window
        self.use_curriculum_learning,self.cl_decay_steps = use_curriculum_learning,cl_decay_steps
        graphs = supports(metadata.adjacency.copy(),filter_type)
        self.encoder_model = DiffusionStack(self.num_nodes,max_diffusion_step,graphs,metadata.feature_dim,rnn_units,num_rnn_layers)
        self.decoder_model = DiffusionStack(self.num_nodes,max_diffusion_step,graphs,self.output_dim,rnn_units,num_rnn_layers,self.output_dim)

    def sampling_threshold(self, batches_seen):
        exponent = batches_seen/self.cl_decay_steps
        return self.cl_decay_steps/(self.cl_decay_steps+math.exp(exponent)) if exponent<700 else 0.

    def _run(self, batch, targets=None, batches_seen=0):
        history = batch['X']
        count = history.shape[0]
        hidden = None
        for inputs in history.unbind(1):
            _,hidden = self.encoder_model(inputs.reshape(count,-1),hidden)
        inputs = history.new_zeros(count,self.num_nodes*self.output_dim)
        outputs = []
        for index in range(self.output_window):
            inputs,hidden = self.decoder_model(inputs,hidden)
            outputs.append(inputs.reshape(count,self.num_nodes,self.output_dim))
            if targets is not None and self.use_curriculum_learning:
                if np.random.uniform(0,1)<self.sampling_threshold(batches_seen):
                    inputs = targets[:,index,:,:self.output_dim].reshape(count,-1)
        return torch.stack(outputs,dim=1)

    def forward(self, batch):
        return self._run(batch)

    def forward_with_teacher(self, batch, targets, batches_seen):
        return self._run(batch,targets,batches_seen)


def curriculum_step(loss_fn):
    def step(model,batch,state):
        prediction = model.forward_with_teacher({'X':batch['X']},batch['y'],state.batches_seen)
        state.task_state['curriculum_batches_seen'] = state.batches_seen
        return loss_fn(prediction,batch['y'],batch.get('y_raw'))
    return step
