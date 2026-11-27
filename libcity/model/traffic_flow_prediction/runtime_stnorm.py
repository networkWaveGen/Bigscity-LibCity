"""Spatial/temporal normalization with safe singleton statistics."""
import torch
from torch import nn
from torch.nn import functional as F


class SNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, source):
        variance = source.var(2, keepdim=True, unbiased=source.shape[2] > 1)
        normalized = (source-source.mean(2, keepdim=True))/torch.sqrt(variance+1e-5)
        return normalized*self.gamma[None, :, None, None]+self.beta[None, :, None, None]


class TNorm(nn.Module):
    def __init__(self, nodes, channels, momentum=.1):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, nodes, 1))
        self.momentum = momentum

    def forward(self, source):
        if self.training:
            mean = source.mean((0, 3), keepdim=True)
            variance = source.var((0, 3), keepdim=True, unbiased=False)
            count = source.shape[0]*source.shape[3]
            correction = count/(count-1) if count > 1 else 1.
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach().to(self.running_mean.dtype), self.momentum)
                self.running_var.lerp_((variance.detach()*correction).to(self.running_var.dtype), self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        return (source-mean)/torch.sqrt(variance+1e-5)*self.gamma+self.beta


class STNorm(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, blocks=1, layers=4,
                 kernel_size=2, hidden_channels=16, snorm_bool=True, tnorm_bool=True):
        super().__init__()
        if min(blocks, layers, hidden_channels) < 1 or kernel_size < 2:
            raise ValueError('STNorm requires positive blocks/layers/channels and kernel_size >= 2')
        self.output_window, self.output_dim = output_window, metadata.output_dim
        self.snorm_bool, self.tnorm_bool = snorm_bool, tnorm_bool
        self.filter_convs, self.gate_convs, self.residual_convs, self.skip_convs = (nn.ModuleList() for _ in range(4))
        self.sn, self.tn = nn.ModuleList(), nn.ModuleList()
        self.start_conv = nn.Conv2d(metadata.feature_dim, hidden_channels, 1)
        count = 1+int(snorm_bool)+int(tnorm_bool)
        self.receptive_field = 1
        for _ in range(blocks):
            for index in range(layers):
                dilation = 2**index
                if snorm_bool:
                    self.sn.append(SNorm(hidden_channels))
                if tnorm_bool:
                    self.tn.append(TNorm(metadata.num_nodes, hidden_channels))
                for convolutions in (self.filter_convs, self.gate_convs):
                    convolutions.append(nn.Conv2d(count*hidden_channels, hidden_channels,
                        (1, kernel_size), dilation=dilation))
                self.residual_convs.append(nn.Conv2d(hidden_channels, hidden_channels, 1))
                self.skip_convs.append(nn.Conv2d(hidden_channels, hidden_channels, 1))
                self.receptive_field += (kernel_size-1)*dilation
        self.end_conv_1 = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.end_conv_2 = nn.Conv2d(hidden_channels, output_window*metadata.output_dim, 1)

    def forward(self, batch):
        source = batch['X'].transpose(1, 3)
        hidden = self.start_conv(F.pad(source, (max(0, self.receptive_field-source.shape[-1]), 0)))
        skip = None
        for index in range(len(self.filter_convs)):
            residual = hidden
            parts = [hidden]
            if self.tnorm_bool:
                parts.append(self.tn[index](hidden))
            if self.snorm_bool:
                parts.append(self.sn[index](hidden))
            combined = torch.cat(parts, dim=1)
            hidden = torch.tanh(self.filter_convs[index](combined))*torch.sigmoid(self.gate_convs[index](combined))
            contribution = self.skip_convs[index](hidden)
            skip = contribution if skip is None else skip[..., -contribution.shape[-1]:]+contribution
            hidden = self.residual_convs[index](hidden)+residual[..., -hidden.shape[-1]:]
        prediction = self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))[..., -1]
        return prediction.reshape(source.shape[0], self.output_window, self.output_dim, source.shape[2]).transpose(2, 3)
