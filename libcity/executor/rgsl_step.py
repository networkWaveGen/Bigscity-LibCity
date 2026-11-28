"""Explicit stochastic graph training and per-call deterministic evaluation."""
import torch
from torch import nn

from libcity.executor.evaluate import evaluate_loop
from libcity.executor.train import train_loop


def graph_step(loss_fn,strategy):
    def step(model,batch,state):
        prediction,_ = model.forward_with_aux(loss_fn.inputs(batch),temperature=strategy['temperature'])
        return loss_fn(prediction,batch['y'],batch.get('y_raw'))
    return step


def train(context,*,until_epoch=None):
    return train_loop(context,graph_step(context.loss_fn,context.config['training']['strategy']),until_epoch=until_epoch)


class PreparedGraphForecast(nn.Module):
    def __init__(self,model,basis):
        super().__init__()
        self.model = model
        self.register_buffer('basis',basis)

    def forward(self,inputs):
        return self.model.forecast(inputs['X'],self.basis)


def evaluate_rgsl(model,loader,**kwargs):
    model.eval()
    with torch.no_grad():
        basis = model.inference_basis()
    return evaluate_loop(PreparedGraphForecast(model,basis),loader,**kwargs)
