"""Temporal bilinear attention with explicit axes and optional block boundaries."""
import torch
from torch import nn

from libcity.model.runtime_norm import SingletonBatchNorm1d


class TemporalAttention(nn.Module):
    def __init__(self, channels, nodes, groups):
        super().__init__()
        length = sum(groups)
        self.conv1 = nn.Conv2d(channels,1,1,bias=False)
        self.conv2 = nn.Conv2d(nodes,1,1,bias=False)
        self.w = nn.Parameter(torch.empty(nodes,channels))
        self.b = nn.Parameter(torch.zeros(length,length))
        self.v = nn.Parameter(torch.empty(length,length))
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.v)
        self.bn = SingletonBatchNorm1d(length)
        mask = torch.zeros(length,length,dtype=torch.bool)
        start = 0
        for width in groups:
            mask[start:start+width,start:start+width] = True
            start += width
        self.register_buffer('mask',mask)

    def forward(self, value):
        left = self.conv1(value.permute(0,1,3,2))[:,0]
        right = self.conv2(value.permute(0,2,1,3))[:,0]
        logits = self.v@torch.sigmoid(left@self.w@right+self.b)
        logits = self.bn(logits.transpose(1,2)).transpose(1,2)
        return logits.masked_fill(~self.mask,float('-inf')).softmax(-1)
