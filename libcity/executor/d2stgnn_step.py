"""Epoch-based curriculum is explicit and resumes with the shared TrainState."""
from libcity.executor.train import train_loop


def curriculum_step(loss_fn,strategy):
    def step(model,batch,state):
        prediction=model(loss_fn.inputs(batch))
        horizon=(min(prediction.shape[1],state.epoch//strategy['cl_decay_epochs']+1)
            if strategy['use_curriculum_learning'] else prediction.shape[1])
        state.task_state['d2stgnn_curriculum_horizon']=horizon
        raw=batch.get('y_raw')
        return loss_fn(prediction[:,:horizon],batch['y'][:,:horizon],None if raw is None else raw[:,:horizon])
    return step


def train(context,*,until_epoch=None):
    return train_loop(context,curriculum_step(context.loss_fn,context.config['training']['strategy']),until_epoch=until_epoch)
