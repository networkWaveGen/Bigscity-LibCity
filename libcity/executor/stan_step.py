"""Shared sampled softmax with unique positives and bounded negative selection."""
import torch
from torch.nn import functional as F

from libcity.executor.evaluate import StepResult
from libcity.executor.task_evaluation import RankingLoss


class STANLoss(RankingLoss):
    def __init__(self,bundle,config):
        super().__init__(bundle,config)
        self.negative_samples = config['negative_samples']

    def candidates(self,target):
        positives,labels = torch.unique(target,sorted=True,return_inverse=True)
        available = torch.ones(self.output_dim,dtype=torch.bool,device=target.device)
        available[positives] = False
        available = available.nonzero().flatten()
        count = min(self.negative_samples,len(available))
        if count:
            available = available[torch.randperm(len(available),device=target.device)[:count]]
        return torch.cat((positives,available[:count])),labels


def sampled_step(loss_fn):
    def step(model,batch,state):
        candidates,labels = loss_fn.candidates(batch['y'])
        scores = model.forward_candidates(loss_fn.inputs(batch),candidates)
        loss = F.cross_entropy(scores.float() if scores.dtype in (torch.float16,torch.bfloat16) else scores,labels)
        return StepResult(loss,len(labels))
    return step
