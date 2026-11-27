"""The default loop owns updates; custom steps own only their scalar objective."""
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
import math

import torch

from libcity.executor.components import EarlyStopper, MetricTracker, build_optimizer, build_scheduler
from libcity.executor.evaluate import ForecastLoss, evaluate_loop, move_batch, supervised_step
from libcity.utils.runtime_checkpoint import CheckpointManager


@dataclass
class TrainState:
    stage: str = 'train'
    epoch: int = 0
    batches_seen: int = 0
    optimizer_steps: int = 0
    best_metric: float | None = None
    task_state: dict = field(default_factory=dict)


@dataclass
class TrainingContext:
    model: object
    bundle: object
    config: dict
    device: torch.device
    directory: object
    tracker: MetricTracker
    checkpoints: CheckpointManager
    state: TrainState
    optimizer: object
    schedule: object
    amp_scaler: object
    stopper: EarlyStopper
    loss_fn: object
    evaluate_fn: object = None
    native_cpu_amp_backward: bool = False


@lru_cache(maxsize=1)
def cpu_amp_convolution_backward_supported():
    """Probe the actual backend without advancing the experiment's random state."""
    with torch.random.fork_rng(devices=[]):
        layer = torch.nn.Conv2d(1, 1, 1)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            prediction = layer(torch.ones(2, 1, 2, 2))
        try:
            prediction.float().square().mean().backward()
        except RuntimeError as error:
            if 'does not support bf16/f16 backward' in str(error):
                return False
            raise
    return True


def build_context(model, bundle, config, device, directory, logger, *, steps_per_epoch=None, loss_fn=None, evaluate_fn=None):
    training = config['training']
    optimizer = build_optimizer(model, training['optimizer'])
    updates = steps_per_epoch or math.ceil(len(bundle.train) / training['accumulation_steps'])
    schedule = build_scheduler(optimizer, training['scheduler'], epochs=training['epochs'], steps_per_epoch=updates)
    if training['amp'] and device.type == 'cpu' and training['amp_dtype'] != 'bfloat16':
        raise ValueError('CPU AMP requires bfloat16')
    scaler = torch.amp.GradScaler('cuda', enabled=training['amp'] and device.type == 'cuda')
    stopper = EarlyStopper(training['direction'], **training['early_stopping'])
    loss = loss_fn if loss_fn is not None else ForecastLoss(training['loss'], bundle.scaler, bundle.metadata.output_dim)
    tracker = MetricTracker(directory, logger, tensorboard=config['runtime']['tensorboard'])
    convolutions = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                    torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)
    native_cpu_backward = (training['amp'] and device.type=='cpu' and torch.backends.mkldnn.enabled
        and any(isinstance(module, convolutions) for module in model.modules())
        and not cpu_amp_convolution_backward_supported())
    if native_cpu_backward:
        logger.info('CPU AMP: oneDNN bfloat16 convolution backward is unavailable; using native backward kernels')
    return TrainingContext(model, bundle, config, device, directory, tracker,
        CheckpointManager(directory / 'checkpoints'), TrainState(), optimizer, schedule, scaler, stopper, loss,
        evaluate_fn, native_cpu_backward)


def autocast_context(context):
    training = context.config['training']
    if not training['amp']:
        return nullcontext()
    return torch.autocast(context.device.type, dtype=getattr(torch, training['amp_dtype']))


def backward(context, weighted_loss):
    """Keep AMP enabled while using native CPU convolution backward if required."""
    backend = torch.backends.mkldnn.flags(enabled=False) if context.native_cpu_amp_backward else nullcontext()
    with backend:
        context.amp_scaler.scale(weighted_loss).backward()


def optimizer_update(context, divisor):
    """Perform one real update; schedule only if AMP actually updated parameters."""
    if divisor <= 0:
        context.optimizer.zero_grad(set_to_none=True)
        return False
    context.amp_scaler.unscale_(context.optimizer)
    for param in context.model.parameters():
        if param.grad is not None:
            param.grad.div_(divisor)
    clip = context.config['training']['clip_grad_norm']
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(context.model.parameters(), clip)
    previous_scale = context.amp_scaler.get_scale()
    context.amp_scaler.step(context.optimizer)
    context.amp_scaler.update()
    updated = context.amp_scaler.get_scale() >= previous_scale
    context.optimizer.zero_grad(set_to_none=True)
    if updated:
        context.state.optimizer_steps += 1
        context.schedule.step('step')
    return updated


def complete_epoch(context, metrics, training_loss):
    training = context.config['training']
    monitor = training['monitor']
    if monitor not in metrics:
        raise ValueError(f'Training monitor {monitor!r} not reported')
    context.schedule.step('epoch')
    context.schedule.step('metric', metrics)
    improved = context.stopper.update(metrics[monitor])
    context.state.best_metric = context.stopper.best
    if improved:
        context.checkpoints.save_best(context.model)
    context.tracker.record(context.state, dict(train_loss=training_loss,
        lr=context.optimizer.param_groups[0]['lr'], **metrics))
    context.checkpoints.save_last(context)
    return context.stopper.stopped


def train_loop(context, step_fn=None, *, until_epoch=None, epochs=None):
    training = context.config['training']
    if context.bundle.train is None or context.bundle.val is None:
        raise ValueError('Training requires both train and validation loaders')
    step_fn = step_fn or supervised_step(context.loss_fn)
    accumulation = training['accumulation_steps']
    target = training['epochs'] if epochs is None else epochs
    end = target if until_epoch is None else min(target, until_epoch)
    if context.stopper.stopped:
        end = context.state.epoch
    for epoch in range(context.state.epoch, end):
        context.model.train()
        context.optimizer.zero_grad(set_to_none=True)
        group_weight = 0
        total, count = 0., 0
        for index, batch in enumerate(context.bundle.train):
            batch = move_batch(batch, context.device)
            with autocast_context(context):
                result = step_fn(context.model, batch, context.state)
            if not torch.isfinite(result.loss):
                raise ValueError('Training loss must be finite')
            # Weighted micro-batches give the same gradient as their combined
            # objective, including missing targets and a short last batch.
            backward(context, result.loss * result.weight)
            group_weight += result.weight
            total += result.loss.item() * result.weight
            count += result.weight
            context.state.batches_seen += 1
            if (index + 1) % accumulation == 0 or index + 1 == len(context.bundle.train):
                optimizer_update(context, group_weight)
                group_weight = 0
        evaluator = context.evaluate_fn or evaluate_loop
        metrics = evaluator(context.model, context.bundle.val, device=context.device,
            loss_fn=context.loss_fn, evaluation=context.config['evaluation'], prefix='val_').metrics
        context.state.epoch = epoch + 1
        if complete_epoch(context, metrics, total / count if count else 0.):
            break
    if context.checkpoints.best_state is None:
        raise ValueError('No completed epoch with a finite validation metric is available')
    # last.pt remains the most recent complete training state, independently
    # of the selected inference weights restored for final test evaluation.
    context.model.load_state_dict(context.checkpoints.best_state)
    return float(context.state.best_metric)
