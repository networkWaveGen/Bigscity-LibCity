"""Native dense and graph-recurrent forecasts sharing the common training loop."""
import numpy as np
import torch
from torch import nn

from libcity.model.traffic_flow_prediction.AGCRN import AVWDCRNN
from libcity.model.traffic_speed_prediction.TGCN import calculate_normalized_laplacian
from libcity.model.traffic_speed_prediction.ATDM import AgnosticConvBlock, ConvRegBlock


class AutoEncoder(nn.Module):
    def __init__(self, *, metadata, input_window, output_window):
        super().__init__()
        self.num_nodes, self.output_dim, self.output_window = metadata.num_nodes, metadata.output_dim, output_window
        self.encoder = nn.Sequential(nn.Linear(input_window*metadata.num_nodes*metadata.feature_dim, 64),
                                     nn.ReLU(), nn.Linear(64, 16))
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(),
                                     nn.Linear(64, output_window*metadata.num_nodes*metadata.output_dim))

    def forward(self, batch):
        prediction = self.decoder(self.encoder(batch['X'].flatten(1)))
        return prediction.reshape(-1, self.output_window, self.num_nodes, self.output_dim)


class AGCRN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, rnn_units=64, embed_dim=10,
                 num_layers=2, cheb_order=2):
        super().__init__()
        if min(rnn_units, embed_dim, num_layers) < 1 or cheb_order < 2:
            raise ValueError('AGCRN dimensions/layers must be positive and cheb_order >= 2')
        self.num_nodes, self.output_dim = metadata.num_nodes, metadata.output_dim
        self.hidden_dim, self.num_layers, self.output_window = rnn_units, num_layers, output_window
        self.node_embeddings = nn.Parameter(torch.empty(metadata.num_nodes, embed_dim))
        self.encoder = AVWDCRNN(dict(num_nodes=metadata.num_nodes, feature_dim=metadata.feature_dim,
            rnn_units=rnn_units, embed_dim=embed_dim, num_layers=num_layers, cheb_order=cheb_order))
        self.end_conv = nn.Conv2d(1, output_window*metadata.output_dim, (1, rnn_units))
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def forward(self, batch):
        source = batch['X']
        state = source.new_zeros(self.num_layers, source.shape[0], self.num_nodes, self.hidden_dim)
        output, _ = self.encoder(source, state, self.node_embeddings)
        output = self.end_conv(output[:, -1:]).squeeze(-1)
        return output.reshape(-1, self.output_window, self.output_dim, self.num_nodes).transpose(2, 3)


class TGCNCell(nn.Module):
    def __init__(self, adjacency, input_dim, hidden):
        super().__init__()
        graph = calculate_normalized_laplacian(np.array(adjacency, copy=True)).tocoo()
        self.register_buffer('normalized_adj', torch.sparse_coo_tensor(
            np.stack((graph.row, graph.col)), graph.data, graph.shape, dtype=torch.float32).coalesce())
        for index, width in enumerate((2*hidden, hidden)):
            weight = nn.Parameter(torch.empty(input_dim+hidden, width))
            nn.init.xavier_normal_(weight)
            self.register_parameter(f'weights_{index}', weight)
            self.register_parameter(f'bias_{index}', nn.Parameter(torch.zeros(width)))

    def graph_project(self, inputs, state, weight, bias):
        features = torch.cat((inputs, state), dim=-1)
        batch, nodes, width = features.shape
        aggregate = torch.sparse.mm(self.normalized_adj, features.permute(1, 2, 0).reshape(nodes, -1))
        aggregate = aggregate.reshape(nodes, width, batch).permute(2, 0, 1)
        return aggregate @ weight + bias

    def forward(self, inputs, state):
        reset, update = torch.sigmoid(self.graph_project(inputs, state, self.weights_0, self.bias_0)).chunk(2, dim=-1)
        candidate = torch.tanh(self.graph_project(inputs, reset*state, self.weights_1, self.bias_1))
        return update*state + (1-update)*candidate


class TGCN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, rnn_units=64, regularization=.0015):
        super().__init__()
        if rnn_units < 1 or regularization < 0:
            raise ValueError('TGCN requires positive rnn_units and nonnegative regularization')
        self.num_nodes, self.hidden = metadata.num_nodes, rnn_units
        self.output_dim, self.output_window, self.regularization = metadata.output_dim, output_window, regularization
        self.tgcn_model = TGCNCell(metadata.adjacency, metadata.feature_dim, rnn_units)
        self.output_model = nn.Linear(rnn_units, output_window*metadata.output_dim)

    def forward(self, batch):
        source = batch['X']
        state = source.new_zeros(source.shape[0], self.num_nodes, self.hidden)
        for step in source.unbind(dim=1):
            state = self.tgcn_model(step, state)
        prediction = self.output_model(state).reshape(-1, self.num_nodes, self.output_window, self.output_dim)
        return prediction.transpose(1, 2)


def tgcn_step(loss_fn):
    from libcity.executor.evaluate import StepResult
    def step(model, batch, state):
        result = loss_fn(model(loss_fn.inputs(batch)), batch['y'], batch.get('y_raw'))
        regularizer = sum(parameter.square().sum() for parameter in model.parameters())
        objective = .5*result.loss + .5*model.regularization*regularizer/max(result.weight, 1)
        return StepResult(objective, result.weight)
    return step


class SpatialAgnosticNetwork(nn.Module):
    def __init__(self, metadata, history, horizon, past_t, hidden, dropout):
        super().__init__()
        self.past_t, self.horizon, self.outputs = past_t, horizon, metadata.output_dim
        self.conv_block = AgnosticConvBlock(metadata.feature_dim, metadata.num_nodes, past_t, hidden, 1)
        self.convT = nn.ConvTranspose2d(hidden, hidden, (1, metadata.num_nodes))
        self.drop = nn.Dropout2d(dropout)
        self.regressor = ConvRegBlock(history, horizon*metadata.output_dim, metadata.num_nodes, hidden)

    def forward(self, source):
        padded = torch.nn.functional.pad(source, (0, 0, self.past_t-1, 0))
        hidden = self.drop(self.convT(self.conv_block(padded)))
        prediction = self.regressor(hidden.flatten(1, 2))
        return prediction.reshape(source.shape[0], self.outputs, self.horizon, source.shape[-1])


class ATDM(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, hidden_size=64, past_t=3, dropout=.2):
        super().__init__()
        if min(hidden_size, past_t) < 1:
            raise ValueError('ATDM hidden_size and past_t must be positive')
        self.sann = SpatialAgnosticNetwork(metadata, input_window, output_window, past_t, hidden_size, dropout)

    def forward(self, batch):
        return self.sann(batch['X'].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
