"""Gumbel/teacher sampling and graph-prior schedule over the shared train loop."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.executor.evaluate import StepResult,evaluate_loop
from libcity.executor.curriculum import teacher_probability
from libcity.executor.train import train_loop


def graph_step(loss_fn, strategy):
    def step(model,batch,state):
        probability = teacher_probability(state.batches_seen,strategy['cl_decay_steps']) if strategy['use_curriculum_learning'] else 0.
        # Keep one NumPy curriculum draw per forecast step, as in the original.
        choices = [bool(np.random.uniform()<probability) for _ in range(model.output_window)] if strategy['use_curriculum_learning'] else None
        prediction,logits = model.forward_with_aux(loss_fn.inputs(batch),targets=batch['y'] if choices is not None else None,
            teacher_mask=choices,temperature=strategy['temperature'])
        forecast = loss_fn(prediction,batch['y'],batch.get('y_raw'))
        active = state.epoch<strategy['graph_regularization_epochs'] and strategy['graph_regularization_weight']>0
        objective,weight = forecast.loss,forecast.weight
        if active:
            logits = logits.float() if logits.dtype in (torch.float16,torch.bfloat16) else logits
            objective = objective+strategy['graph_regularization_weight']*F.binary_cross_entropy_with_logits(logits,model.graph_prior)
            weight = weight or prediction.numel()
        state.task_state.update(curriculum_batches_seen=state.batches_seen,graph_regularization_epoch=state.epoch,
            graph_regularization_active=active)
        return StepResult(objective,weight)
    return step


def train(context,*,until_epoch=None):
    return train_loop(context,graph_step(context.loss_fn,context.config['training']['strategy']),until_epoch=until_epoch)


class PreparedGraphForecast(nn.Module):
    def __init__(self, model, adjacency):
        super().__init__()
        self.model = model
        self.register_buffer('adjacency',adjacency)

    def forward(self, inputs):
        return self.model.forecast(inputs['X'],self.adjacency)


def evaluate_gts(model,loader,**kwargs):
    model.eval()
    with torch.no_grad():
        adjacency = model.inference_graph()
    # Scope the prepared graph to this evaluation. It cannot survive a later
    # optimizer update, checkpoint load or device/dtype change on the model.
    return evaluate_loop(PreparedGraphForecast(model,adjacency),loader,**kwargs)
