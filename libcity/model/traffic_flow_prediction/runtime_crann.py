"""CRANN consumes historical observations and explicitly known future covariates."""
import torch
from torch import nn

from libcity.model.traffic_flow_prediction.CRANN import ConvBlock, AttentionBlock, MLP


class TemporalEncoder(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)

    def forward(self, source):
        # PyTorch initializes hidden states on the input's device and dtype.
        return self.lstm(source)


class TemporalDecoder(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.fc_hidden = nn.Linear(hidden, hidden, bias=False)
        self.fc_encoder = nn.Linear(hidden, hidden, bias=False)
        self.weight = nn.Parameter(torch.empty(1, hidden))
        nn.init.xavier_uniform_(self.weight)
        self.attn_combine = nn.Linear(hidden*2, hidden)
        self.lstm = nn.LSTM(hidden+1, hidden, layers, batch_first=True)
        self.fc_prediction = nn.Linear(hidden, 1)

    def forward(self, inputs, hidden, encoded):
        scores = torch.tanh(self.fc_hidden(hidden[0][-1]).unsqueeze(1) + self.fc_encoder(encoded))
        weights = torch.softmax(scores @ self.weight.T, dim=1)
        context = (weights*encoded).sum(dim=1)
        output, hidden = self.lstm(torch.cat((inputs, context), dim=-1).unsqueeze(1), hidden)
        return self.fc_prediction(output[:, 0]), hidden


class SpatialAttention(nn.Module):
    def __init__(self, history, horizon, rows, columns):
        super().__init__()
        self.conv_block1 = ConvBlock(history, 64, 5)
        self.regressor = nn.Conv2d(64, horizon, kernel_size=3, padding=1)
        # The old implementation hard-coded 24 even for a different horizon.
        self.att1 = AttentionBlock(rows, columns, horizon, method='hadamard')

    def forward(self, source):
        return self.att1(self.regressor(self.conv_block1(source)))[0]


class CRANN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, dim_x=5, dim_y=6,
                 n_hidden_tem=100, n_layers_tem=1, n_hidden_dns=0, n_layers_dns=1,
                 n_ar=4, temporal_channel=1):
        super().__init__()
        if metadata.input_layout.get('kind') != 'crann':
            raise ValueError('CRANN requires data.sampling.name=crann')
        if min(dim_x, dim_y, n_hidden_tem, n_layers_tem, n_layers_dns, n_ar) < 1:
            raise ValueError('CRANN dimensions, layers and n_ar must be positive')
        if dim_x*dim_y != metadata.num_nodes or metadata.output_dim != 1:
            raise ValueError('CRANN requires dim_x*dim_y == num_nodes and a scalar target')
        if not 0 <= temporal_channel < metadata.feature_dim or n_ar > input_window:
            raise ValueError('CRANN requires a valid temporal_channel and n_ar <= input_window')
        if n_layers_dns > 1 and n_hidden_dns < 1:
            raise ValueError('CRANN multilayer dense module requires n_hidden_dns > 0')
        self.dim_x, self.dim_y, self.num_nodes = dim_x, dim_y, metadata.num_nodes
        self.input_window, self.output_window, self.n_ar = input_window, output_window, n_ar
        self.temporal_channel = temporal_channel
        self.temporal_history = metadata.input_layout['temporal_history']
        self.future_dim = len(metadata.input_layout.get('future_feature_indices', ()))
        self.temporal_encoder = TemporalEncoder(n_hidden_tem, n_layers_tem)
        self.temporal_decoder = TemporalDecoder(n_hidden_tem, n_layers_tem)
        self.spatial_model = SpatialAttention(input_window, output_window, dim_x, dim_y)
        inputs = output_window*(metadata.num_nodes+self.future_dim+1) + n_ar*metadata.num_nodes
        self.mlp = MLP(inputs, output_window*metadata.num_nodes, n_layers_dns, n_hidden_dns)

    def forward(self, batch):
        source = batch['X']
        count = source.shape[0]
        temporal = source[:, -self.temporal_history:, 0, self.temporal_channel:self.temporal_channel+1]
        encoded, hidden = self.temporal_encoder(temporal)
        following = temporal.new_zeros(count, 1)
        predictions = []
        for _ in range(self.output_window):
            following, hidden = self.temporal_decoder(following, hidden, encoded)
            predictions.append(following)
        temporal = torch.stack(predictions, dim=1)
        spatial = source[:, -self.input_window:, :, 0].reshape(count, self.input_window, self.dim_x, self.dim_y)
        spatial_prediction = self.spatial_model(spatial).reshape(count, self.output_window, self.num_nodes)
        features = [temporal, spatial_prediction]
        if self.future_dim:
            features.append(batch['future_features'][:, :, 0])
        combined = torch.cat(features, dim=-1).reshape(count, -1)
        autoregressive = spatial[:, -self.n_ar:].reshape(count, -1)
        return self.mlp(torch.cat((combined, autoregressive), dim=-1)).reshape(
            count, self.output_window, self.num_nodes, 1)
