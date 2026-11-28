"""Python-RNG teacher forcing and exact grouped RMSE over the shared loop."""
import random

from libcity.executor.curriculum import teacher_probability
from libcity.executor.evaluate import ForecastLoss
from libcity.executor.root_loss import RootMeanSquaredLoss
from libcity.executor.train import train_loop


def build_loss(bundle,config):
    return (RootMeanSquaredLoss(bundle,config) if config['name'] in ('rmse','masked_rmse') else
        ForecastLoss(config,bundle.scaler,bundle.metadata.output_dim))


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
