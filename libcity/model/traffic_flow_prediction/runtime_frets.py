"""Frequency-domain forecasting, including multivariate input/output layouts."""
import torch
from torch import nn
from torch.nn import functional as F


class FreTS(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, embed_size=128, hidden_size=256,
                 channel_independence=True, sparsity_threshold=.01, scale=.02):
        super().__init__()
        if min(embed_size, hidden_size) < 1 or sparsity_threshold < 0 or scale <= 0:
            raise ValueError('FreTS requires positive dimensions/scale and nonnegative sparsity_threshold')
        self.seq_length, self.pre_length, self.num_nodes = input_window, output_window, metadata.num_nodes
        self.output_dim, self.channel_independence = metadata.output_dim, channel_independence
        self.sparsity_threshold = sparsity_threshold
        self.embeddings = nn.Parameter(torch.randn(metadata.feature_dim, embed_size))
        for index in (1, 2):
            for name in ('r', 'i'):
                self.register_parameter(f'{name}{index}', nn.Parameter(scale*torch.randn(embed_size, embed_size)))
            for name in ('rb', 'ib'):
                self.register_parameter(f'{name}{index}', nn.Parameter(scale*torch.randn(embed_size)))
        self.fc = nn.Sequential(nn.Linear(input_window*embed_size, hidden_size), nn.LeakyReLU(),
                                nn.Linear(hidden_size, output_window*metadata.output_dim))

    def frequency_mlp(self, source, index):
        real, imaginary = getattr(self, f'r{index}').diagonal(), getattr(self, f'i{index}').diagonal()
        # Preserve the original diagonal "dd" frequency contraction.
        a = F.relu(source.real*real-source.imag*imaginary+getattr(self, f'rb{index}'))
        b = F.relu(source.imag*real+source.real*imaginary+getattr(self, f'ib{index}'))
        pair = F.softshrink(torch.stack((a, b), dim=-1), lambd=self.sparsity_threshold)
        return torch.view_as_complex(pair.contiguous())

    def forward(self, batch):
        source = batch['X']
        embedded = torch.einsum('btnf,fe->bnte', source, self.embeddings)
        # CPU FFT does not accept bfloat16 produced by autocast einsum.
        if embedded.dtype not in (torch.float32, torch.float64):
            embedded = embedded.float()
        hidden = embedded
        if self.channel_independence:
            hidden = hidden.transpose(1, 2)
            hidden = torch.fft.irfft(self.frequency_mlp(torch.fft.rfft(hidden, dim=2, norm='ortho'), 1),
                                     n=self.num_nodes, dim=2, norm='ortho').transpose(1, 2)
        hidden = torch.fft.irfft(self.frequency_mlp(torch.fft.rfft(hidden, dim=2, norm='ortho'), 2),
                                 n=self.seq_length, dim=2, norm='ortho') + embedded
        output = self.fc(hidden.flatten(2))
        return output.reshape(source.shape[0], self.num_nodes, self.pre_length, self.output_dim).transpose(1, 2)
