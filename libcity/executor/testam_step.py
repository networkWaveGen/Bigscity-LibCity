"""Expert error and routing supervision with masks from raw target values."""
import torch
from torch.nn import functional as F

from libcity.executor.evaluate import StepResult


def routing_labels(gate,expert_error,routed_error,valid,*,quantile=None):
    """Labels are probabilities over the last (expert) axis, never batch indices."""
    with torch.no_grad():
        if not valid.any():
            return torch.zeros_like(gate),torch.zeros_like(gate)
        chosen = gate.argmax(dim=-1)
        selected = F.one_hot(chosen,3).to(gate.dtype)
        if quantile is not None:
            observed = routed_error[valid]
            high,low = torch.quantile(observed,quantile),torch.quantile(observed,1-quantile)
            incorrect = (routed_error>high)&valid
            correct = (routed_error<low)&valid
            worst = torch.where(incorrect[...,None],(1-selected)/2,selected)
            best = selected*correct[...,None]
        else:
            maximum,minimum = expert_error.argmax(-1),expert_error.argmin(-1)
            max_onehot = F.one_hot(maximum,3).to(gate.dtype)
            min_onehot = F.one_hot(minimum,3).to(gate.dtype)
            worst = torch.where((maximum==chosen)[...,None],(1-max_onehot)/2,selected)
            best = torch.where((minimum==chosen)[...,None],selected,(1+min_onehot)/4)
        return worst*valid[...,None],best*valid[...,None]


def expert_step(loss_fn):
    def step(model,batch,state):
        prediction,gate,experts = model.forward_with_aux(loss_fn.inputs(batch))
        raw_prediction,target = loss_fn.raw(prediction,batch['y'],batch.get('y_raw'))
        target = target.permute(0,2,1,3)
        valid = torch.isfinite(target)
        if loss_fn.name.startswith('masked_') and loss_fn.mask_value is not None:
            valid &= target!=loss_fn.mask_value
        count = int(valid.sum())
        if not count:
            return StepResult(raw_prediction.permute(0,2,1,3)[valid].sum(),0)
        raw_experts = loss_fn.scaler.inverse_transform(experts.float())
        expanded = valid.unsqueeze(-2).expand_as(raw_experts)
        if not torch.isfinite(raw_experts[expanded]).all():
            raise ValueError('TESTAM expert prediction is nonfinite on valid targets')
        safe_target = torch.where(valid,target,0.)
        safe_experts = torch.where(expanded,raw_experts,0.)
        error = (safe_experts-safe_target.unsqueeze(-2)).abs()
        objective = error.square()[expanded].mean() if loss_fn.name.endswith('mse') else error[expanded].mean()
        valid_gate = valid.any(dim=-1)
        counts = valid.sum(-1).clamp_min(1)
        expert_error = error.sum(-1)/counts.unsqueeze(-1)
        routed = torch.where(valid,raw_prediction.permute(0,2,1,3),0.)
        routed_error = (routed-safe_target).abs().sum(-1)/counts
        worst,best = routing_labels(gate,expert_error,routed_error,valid_gate,
            quantile=model.quantile if model.is_quantile else None)
        log_gate = gate.float().clamp_min(torch.finfo(torch.float32).tiny).log()
        routing = (-.5*(worst+best)*log_gate).sum()/(int(valid_gate.sum())*3)
        return StepResult(objective+routing,count)
    return step
