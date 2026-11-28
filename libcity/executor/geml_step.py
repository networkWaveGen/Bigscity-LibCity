"""Raw OD and marginal objectives over one additive valid-cell denominator."""
import torch

from libcity.executor.evaluate import ForecastLoss,StepResult


class GEMLLoss(ForecastLoss):
    def __init__(self,bundle,config):
        super().__init__(config,bundle.scaler,bundle.metadata.output_dim)
        self.weights = tuple(config[name] for name in ('od_weight','origin_weight','destination_weight'))


def auxiliary_step(loss_fn):
    def step(model,batch,state):
        prediction,origin,destination = model.forward_with_aux(loss_fn.inputs(batch))
        prediction,target = loss_fn.raw(prediction,batch['y'],batch.get('y_raw'))
        finite = torch.isfinite(target)
        valid = finite & (target!=loss_fn.mask_value) if loss_fn.name.startswith('masked_') and loss_fn.mask_value is not None else finite
        count = int(valid.sum())
        if not count:
            return StepResult(prediction[valid].sum(),0)
        if not torch.isfinite(prediction[valid]).all():
            raise ValueError('GEML prediction is nonfinite on valid OD targets')
        objective = loss_fn.weights[0]*(prediction[valid]-target[valid]).square().sum()
        truth = torch.where(finite,target,0.)
        for weight,forecast,axis in ((loss_fn.weights[1],origin,-2),(loss_fn.weights[2],destination,-3)):
            if weight==0:
                continue
            # Sum raw targets: inverse(sum(normalized)) loses (N-1)*offset.
            forecast = forecast.transpose(-3,-2) if axis==-3 else forecast
            forecast = forecast.float() if forecast.dtype in (torch.float16,torch.bfloat16) else forecast
            forecast = loss_fn.scaler.inverse_transform(forecast)
            known = valid & finite.all(axis,keepdim=True)
            residual = forecast.expand_as(target)[known]-truth.sum(axis,keepdim=True).expand_as(target)[known]
            if not torch.isfinite(residual).all():
                raise ValueError('GEML marginal prediction is nonfinite on known targets')
            objective = objective+weight*residual.square().sum()
        return StepResult(objective/count,count)
    return step
