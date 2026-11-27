"""Per-route cumulative relative RMSE, with whole-trip MAPE for validation."""
import torch

from libcity.executor.evaluate import StepResult
from libcity.executor.task_evaluation import ETALoss


class TTPNetLoss(ETALoss):
    def __call__(self,prediction,target,raw_target=None):
        prediction,target = self.raw(prediction,target,raw_target)
        loss = ((prediction-target).abs()/target.abs().clamp_min(self.config['eps'])).mean()
        return StepResult(loss,len(target))

    def cumulative(self,prediction,target,valid):
        if prediction.dtype in (torch.float16,torch.bfloat16): prediction = prediction.float()
        mean,std = self.bundle.metadata.info['statistics']['time_gap']
        prediction = prediction*std+mean
        target = target[:,1:1+prediction.shape[1]]
        if target.shape!=prediction.shape or not valid.any(1).all() or not torch.isfinite(target[valid]).all() or (target[valid]<=0).any():
            raise ValueError('TTPNet cumulative targets must be finite positive elapsed seconds on valid route points')
        target = target.masked_fill(~valid,1.)
        prediction = prediction.masked_fill(~valid,1.)
        residual = (prediction-target)/target.abs().clamp_min(self.config['eps'])
        # vector_norm has a defined zero gradient at an exact fit, unlike sqrt(sum(x**2)).
        values = torch.linalg.vector_norm(residual,dim=1)/valid.sum(1).to(residual.dtype).sqrt()
        return StepResult(values.mean(),len(values))


def cumulative_step(loss_fn):
    def step(model,batch,state):
        _,prediction,valid = model.forward_with_aux(loss_fn.inputs(batch))
        return loss_fn.cumulative(prediction,batch['local_y_raw'],valid)
    return step
