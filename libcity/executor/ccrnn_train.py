"""Python-RNG teacher forcing and exact grouped RMSE over the shared loop."""
import random

from libcity.executor.curriculum import teacher_probability
from libcity.executor.root_loss import build_forecast_loss as build_loss
from libcity.executor.train import train_loop


def curriculum_step(loss_fn,strategy):
    def step(model,batch,state):
        inputs = loss_fn.inputs(batch)
        if strategy['use_curriculum_learning']:
            probability = teacher_probability(state.batches_seen,strategy['cl_decay_steps'])
            # Preserve the original draw for every decoder step, even the
            # final draw whose teacher value has no subsequent prediction.
            choices = [random.random()<probability for _ in range(model.output_window)]
            prediction = model.forward_with_teacher(inputs,batch['y'],choices)
        else:
            prediction = model(inputs)
        state.task_state['curriculum_batches_seen'] = state.batches_seen
        return loss_fn(prediction,batch['y'],batch.get('y_raw'))
    return step


def train(context,*,until_epoch=None):
    return train_loop(context,curriculum_step(context.loss_fn,context.config['training']['strategy']),until_epoch=until_epoch)
