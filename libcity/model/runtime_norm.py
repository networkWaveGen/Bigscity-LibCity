"""Shared normalization behavior for real singleton spatial tail batches."""
from torch import nn
from torch.nn import functional as F


class SingletonBatchNorm2d(nn.BatchNorm2d):
    def forward(self,source):
        if self.training and source.numel()//source.shape[1]==1:
            # One observed value cannot supply a sample variance. Reuse running
            # statistics without fabricating a duplicate or updating counters.
            return F.batch_norm(source,self.running_mean,self.running_var,self.weight,self.bias,
                                training=False,momentum=self.momentum,eps=self.eps)
        return super().forward(source)
