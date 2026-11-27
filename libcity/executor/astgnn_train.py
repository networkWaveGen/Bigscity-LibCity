"""Two explicit ASTGNN stages composed from the same default training loop."""
import math
from libcity.executor.components import build_optimizer,build_scheduler
from libcity.executor.train import train_loop


def configure_fine_tuning(context):
    training = context.config['training']
    strategy = training['strategy']
    optimizer = {**training['optimizer'],'lr':strategy['fine_tune_lr']}
    context.optimizer = build_optimizer(context.model,optimizer)
    context.schedule = build_scheduler(context.optimizer,training['scheduler'],
        epochs=strategy['fine_tune_epochs'],
        steps_per_epoch=math.ceil(len(context.bundle.train)/training['accumulation_steps']))


def prepare_resume(context,state):
    if state['stage']=='fine_tune':
        configure_fine_tuning(context)


def train(context, *,until_teacher_epoch=None,until_fine_epoch=None):
    strategy = context.config['training']['strategy']
    if set(strategy)-{'fine_tune_epochs','fine_tune_lr'}:
        raise ValueError('Unknown ASTGNN training.strategy keys')
    fine_epochs = strategy['fine_tune_epochs']
    if type(fine_epochs) is not int or fine_epochs<0 or strategy['fine_tune_lr']<=0:
        raise ValueError('ASTGNN requires nonnegative fine_tune_epochs and positive fine_tune_lr')
    if context.state.stage=='train':
        context.state.stage='teacher'
    if context.state.stage=='teacher':
        def teacher_step(model,batch,state):
            return context.loss_fn(model.forward_with_teacher({'X':batch['X']},batch['y']),batch['y'],batch.get('y_raw'))
        score = train_loop(context,teacher_step,until_epoch=until_teacher_epoch)
        if until_teacher_epoch is not None and context.state.epoch<context.config['training']['epochs']:
            return score
        if not fine_epochs:
            return score
        configure_fine_tuning(context)
        context.state.stage,context.state.epoch = 'fine_tune',0
        context.state.task_state['completed_stages'] = ['teacher']
        context.state.task_state['optimizer_resets'] = 1
        context.stopper.wait = 0
        context.checkpoints.save_last(context)
    return train_loop(context,epochs=fine_epochs,until_epoch=until_fine_epoch)
