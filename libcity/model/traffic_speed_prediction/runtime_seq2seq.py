"""Encoder-decoder inference is deterministic; teacher forcing is training-only."""
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, recurrent, inputs, hidden, layers, dropout, bidirectional):
        super().__init__()
        self.rnn = recurrent(inputs, hidden, layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, source):
        return self.rnn(source)[1]


class Decoder(nn.Module):
    def __init__(self, recurrent, outputs, hidden, layers, dropout, bidirectional):
        super().__init__()
        self.rnn = recurrent(outputs, hidden, layers, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden*(2 if bidirectional else 1), outputs)

    def forward(self, source, state):
        output, state = self.rnn(source.unsqueeze(0), state)
        return self.fc(output[0]), state


class Seq2Seq(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, rnn_type='GRU', hidden_size=64,
                 num_layers=1, dropout=0., bidirectional=False, teacher_forcing_ratio=0.):
        super().__init__()
        if rnn_type not in ('RNN','LSTM','GRU') or not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError('Seq2Seq requires RNN/LSTM/GRU and teacher_forcing_ratio in [0,1]')
        self.num_nodes, self.output_dim, self.output_window = metadata.num_nodes, metadata.output_dim, output_window
        self.teacher_forcing_ratio = teacher_forcing_ratio
        recurrent = getattr(nn, rnn_type)
        self.encoder = Encoder(recurrent, metadata.num_nodes*metadata.feature_dim, hidden_size, num_layers, dropout, bidirectional)
        self.decoder = Decoder(recurrent, metadata.num_nodes*metadata.output_dim, hidden_size, num_layers, dropout, bidirectional)

    def _decode(self, batch, targets=None):
        source = batch['X']
        count = source.shape[0]
        state = self.encoder(source.transpose(0, 1).flatten(2))
        following = source.new_zeros(count, self.num_nodes*self.output_dim)
        predictions = []
        for index in range(self.output_window):
            following, state = self.decoder(following, state)
            predictions.append(following.reshape(count, self.num_nodes, self.output_dim))
            if targets is not None and torch.rand((), device=source.device) < self.teacher_forcing_ratio:
                following = targets[:, index, :, :self.output_dim].reshape(count, -1)
        return torch.stack(predictions, dim=1)

    def forward(self, batch):
        return self._decode(batch)

    def forward_with_teacher(self, batch, targets):
        return self._decode(batch, targets)


def teacher_step(loss_fn):
    def step(model, batch, state):
        prediction = model.forward_with_teacher(loss_fn.inputs(batch), batch['y'])
        return loss_fn(prediction, batch['y'], batch.get('y_raw'))
    return step
