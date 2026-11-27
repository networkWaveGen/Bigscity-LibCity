"""GRU example: model parameters and static metadata, with no training machinery."""
import torch
from torch import nn


class AutoregressiveRNN(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, hidden_size=64,
                 num_layers=1, dropout=0., bidirectional=False, rnn_type='GRU', teacher_forcing_ratio=0.):
        super().__init__()
        if rnn_type not in ('RNN', 'LSTM', 'GRU'):
            raise ValueError('rnn_type must be RNN, LSTM or GRU')
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError('teacher_forcing_ratio must be in [0,1]')
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.num_nodes, self.feature_dim, self.output_dim = metadata.num_nodes, metadata.feature_dim, metadata.output_dim
        self.input_window, self.output_window = input_window, output_window
        self.future_indices = tuple(metadata.input_layout.get('future_feature_indices', range(self.output_dim,self.feature_dim)))
        self.rnn = getattr(nn, rnn_type)(self.num_nodes * self.feature_dim, hidden_size, num_layers,
                          dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), self.num_nodes * self.output_dim)

    def forward(self, batch):
        return self._decode(batch)

    def forward_with_teacher(self, batch, targets):
        return self._decode(batch, targets)

    def _decode(self, batch, targets=None):
        source = batch['X'].permute(1, 0, 2, 3)
        count = source.shape[1]
        source = source.reshape(self.input_window, count, -1)
        predictions = []
        for step in range(self.output_window):
            output, _ = self.rnn(source)
            prediction = self.fc(output[-1]).reshape(count, self.num_nodes, self.output_dim)
            predictions.append(prediction)
            following = source[-1].reshape(count,self.num_nodes,self.feature_dim).clone()
            following[...,:self.output_dim] = prediction
            if targets is not None and torch.rand((), device=source.device) < self.teacher_forcing_ratio:
                following[...,:self.output_dim] = targets[:, step, :, :self.output_dim]
            if self.future_indices:
                if 'future_features' not in batch:
                    raise ValueError('Autoregressive extra channels require explicit future_features')
                following[...,list(self.future_indices)] = batch['future_features'][:, step]
            source = torch.cat((source[1:], following.reshape(1, count, -1)), dim=0)
        return torch.stack(predictions, dim=1)


class GRU(AutoregressiveRNN):
    """Traffic GRU retains its public name and parameter layout."""
