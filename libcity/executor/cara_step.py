"""Independent BPR pairs and bounded sampling, with deterministic full ranking."""
import torch
from torch.nn import functional as F

from libcity.executor.evaluate import StepResult
from libcity.executor.task_evaluation import TaskLoss


class CARALoss(TaskLoss):
    def sample(self,batch):
        target = batch['y']
        allowed = torch.ones(len(target),self.output_dim,dtype=torch.bool,device=target.device)
        locations = batch['current_loc']
        valid = torch.arange(locations.shape[1],device=target.device)[None]<batch['lengths'][:,None]
        rows = torch.arange(len(target),device=target.device)[:,None].expand_as(locations)
        allowed[rows[valid],locations[valid]-1] = False
        allowed.scatter_(1,target[:,None],False)
        # If every POI has been visited, any different target remains a valid
        # negative. A one-location vocabulary has no ranking pair at all.
        exhausted = ~allowed.any(1)
        allowed[exhausted] = True
        allowed.scatter_(1,target[:,None],False)
        valid = allowed.any(1)
        weights = allowed.float()
        weights[~valid,0] = 1.
        return torch.multinomial(weights,1).flatten(),valid

    @staticmethod
    def stable(scores):
        return scores.float() if scores.dtype in (torch.float16,torch.bfloat16) else scores

    def pairs(self,selected,valid):
        if not valid.any():
            return StepResult(selected.sum()*0.,0)
        values = self.stable(selected[valid])
        return StepResult((1+F.softplus(values[:,1]-values[:,0])).mean(),int(valid.sum()))

    def __call__(self,scores,target,raw_target=None):
        values = self.stable(scores)
        if values.shape[1]==1:
            return StepResult(values.sum()*0.,len(target))
        positive = values.gather(1,target[:,None])
        losses = (1+F.softplus(values-positive)).scatter(1,target[:,None],0.)
        return StepResult((losses.sum(-1)/(values.shape[1]-1)).mean(),len(target))


def pairwise_step(loss_fn):
    def step(model,batch,state):
        negative,valid = loss_fn.sample(batch)
        candidates = torch.stack((batch['y'],negative),dim=-1)
        selected = model.forward_candidates(loss_fn.inputs(batch),candidates)
        return loss_fn.pairs(selected,valid)
    return step
