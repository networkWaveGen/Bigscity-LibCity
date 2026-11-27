"""GRU example: model parameters and static metadata, with no training machinery."""
import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, hidden_size=64,
                 num_layers=1, dropout=0., bidirectional=False):
        super().__init__()
        self.num_nodes, self.feature_dim, self.output_dim = metadata.num_nodes, metadata.feature_dim, metadata.output_dim
        self.input_window, self.output_window = input_window, output_window
        self.rnn = nn.GRU(self.num_nodes * self.feature_dim, hidden_size, num_layers,
                          dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), self.num_nodes * self.output_dim)

    def forward(self, batch):
        source = batch['X'].permute(1, 0, 2, 3)
        count = source.shape[1]
        source = source.reshape(self.input_window, count, -1)
        predictions = []
        for step in range(self.output_window):
            output, _ = self.rnn(source)
            prediction = self.fc(output[-1]).reshape(count, self.num_nodes, self.output_dim)
            predictions.append(prediction)
            following = prediction
            if self.feature_dim > self.output_dim:
                if 'future_features' not in batch:
                    raise ValueError('Autoregressive extra channels require explicit future_features')
                following = torch.cat((prediction, batch['future_features'][:, step]), dim=-1)
            source = torch.cat((source[1:], following.reshape(1, count, -1)), dim=0)
        return torch.stack(predictions, dim=1)
