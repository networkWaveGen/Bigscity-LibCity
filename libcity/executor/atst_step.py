"""Sampled normalized BPR training; deterministic full-candidate validation."""
import torch
from torch.nn import functional as F

from libcity.executor.evaluate import StepResult
from libcity.executor.task_evaluation import TaskLoss


class ATSTRankingLoss(TaskLoss):
    def __init__(self,bundle,config):
        super().__init__(bundle,config)
        self.weights = torch.tensor(bundle.metadata.info['negative_weights'].copy(),dtype=torch.float32)
        self.negative_samples = min(config['negative_samples'],int((self.weights>0).sum())-1)

    @staticmethod
    def normalized(scores):
        # Avoid BF16 norm underflow while preserving an explicit float64 caller.
        return F.normalize(scores.float() if scores.dtype in (torch.float16,torch.bfloat16) else scores,dim=-1)

    def sample(self,target,device):
        weights = self.weights.to(device)[None].expand(len(target),-1).clone()
        weights.scatter_(1,target[:,None],0.)
        return torch.multinomial(weights,self.negative_samples,replacement=False)

    def sampled(self,scores,target,negatives):
        chosen = torch.cat((target[:,None],negatives),dim=-1)
        return self.selected(scores.gather(1,chosen))

    def selected(self,scores):
        values = self.normalized(scores)
        loss = F.softplus(values[:,1:]-values[:,:1]).sum(-1).mean()
        return StepResult(loss,len(scores))

    def __call__(self,scores,target,raw_target=None):
        values = self.normalized(scores)
        positive = values.gather(1,target[:,None])
        losses = F.softplus(values-positive)
        losses = losses.scatter(1,target[:,None],0.)
        return StepResult(losses.sum(-1).mean(),len(target))


def pairwise_step(loss_fn):
    def step(model,batch,state):
        negatives = loss_fn.sample(batch['y'],batch['y'].device)
        candidates = torch.cat((batch['y'][:,None],negatives),dim=-1)
        scores = model.forward_candidates(loss_fn.inputs(batch),candidates)
        return loss_fn.selected(scores)
    return step
