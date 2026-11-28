"""Explicit block masks and valid-target-weighted latent consistency training."""
import math

import torch

from libcity.executor.evaluate import ForecastLoss,StepResult
from libcity.executor.train import train_loop


class SSTBANLoss(ForecastLoss):
    def __init__(self,bundle,config):
        super().__init__(config,bundle.scaler,bundle.metadata.output_dim)
        self.self_weight=config['self_weight']


def block_keep_mask(shape,length,probability,device):
    batch,steps,nodes,hidden=shape
    kept=torch.rand(batch,math.ceil(steps/length),nodes,hidden,device=device)<probability
    return kept.repeat_interleave(length,dim=1)[:,:steps]


def consistency_cost(complete,corrupted):
    with torch.autocast(complete.device.type,enabled=False):
        if complete.dtype in (torch.float16,torch.bfloat16): complete,corrupted=complete.float(),corrupted.float()
        if not torch.isfinite(complete).all() or not torch.isfinite(corrupted).all():
            raise ValueError('SSTBAN auxiliary representations must be finite')
        valid=corrupted!=0
        squared=torch.where(valid,(complete-corrupted).square(),torch.zeros_like(complete)).flatten(1)
        count=valid.flatten(1).sum(-1)
        return squared.sum(-1)/count.clamp_min(1)


def auxiliary_step(loss_fn,strategy):
    def step(model,batch,state):
        inputs=loss_fn.inputs(batch)
        if loss_fn.self_weight==0:
            return loss_fn(model(inputs),batch['y'],batch.get('y_raw'))
        mask=block_keep_mask((len(batch['X']),model.steps,model.nodes,model.hidden_dim),
            strategy['block_length'],strategy['keep_probability'],batch['X'].device)
        prediction,complete,corrupted=model.forward_with_aux(inputs,mask)
        main=loss_fn(prediction,batch['y'],batch.get('y_raw'))
        if not main.weight: return main
        _,target=loss_fn.raw(prediction,batch['y'],batch.get('y_raw'))
        valid=torch.isfinite(target)
        if loss_fn.name.startswith('masked_') and loss_fn.mask_value is not None: valid&=target!=loss_fn.mask_value
        dtype=torch.float32 if complete.dtype in (torch.float16,torch.bfloat16) else complete.dtype
        counts=valid.flatten(1).sum(-1).to(dtype)
        auxiliary=(consistency_cost(complete,corrupted)*counts).sum()/main.weight
        weight=loss_fn.self_weight
        return StepResult((1-weight)*main.loss+weight*auxiliary,main.weight)
    return step


def train(context,*,until_epoch=None):
    return train_loop(context,auxiliary_step(context.loss_fn,context.config['training']['strategy']),until_epoch=until_epoch)
