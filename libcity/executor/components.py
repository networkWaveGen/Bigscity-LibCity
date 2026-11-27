"""Independent optimizer, scheduler, early-stop and recording tools."""
from dataclasses import asdict, dataclass
import json
import math

import torch


def build_optimizer(model, config):
    options = dict(config)
    name = options.pop('name')
    declarations = options.pop('groups', [])
    named = dict(model.named_parameters())
    assigned, groups = set(), []
    for declaration in declarations:
        declaration = dict(declaration)
        modules = declaration.pop('modules')
        if isinstance(modules, str):
            modules = [modules]
        selected = [key for key in named if any(key == prefix or key.startswith(prefix + '.') for prefix in modules)]
        if not selected:
            raise ValueError(f'Optimizer group matches no parameters: {modules}')
        if assigned.intersection(selected):
            raise ValueError('Optimizer parameter groups overlap')
        assigned.update(selected)
        frozen = declaration.pop('freeze', False)
        if frozen:
            for key in selected:
                named[key].requires_grad_(False)
        else:
            groups.append(dict(params=[named[key] for key in selected if named[key].requires_grad], **declaration))
    remaining = [param for key, param in named.items() if key not in assigned and param.requires_grad]
    if remaining:
        groups.append({'params': remaining})
    if not any(group['params'] for group in groups):
        raise ValueError('Optimizer has no trainable parameters')
    constructors = dict(adam=torch.optim.Adam, adamw=torch.optim.AdamW, sgd=torch.optim.SGD,
                        rmsprop=torch.optim.RMSprop, adagrad=torch.optim.Adagrad)
    options.setdefault('lr', .001)
    return constructors[name](groups, **options)


@dataclass
class Schedule:
    scheduler: object
    interval: str
    monitor: str = 'val_loss'

    def step(self, interval, metrics=None):
        if self.scheduler is None or self.interval != interval:
            return
        if interval == 'metric':
            if self.monitor not in metrics:
                raise ValueError(f'Scheduler metric {self.monitor!r} not reported')
            self.scheduler.step(metrics[self.monitor])
        else:
            self.scheduler.step()

    def state_dict(self):
        return None if self.scheduler is None else self.scheduler.state_dict()

    def load_state_dict(self, state):
        if (state is None) != (self.scheduler is None):
            raise ValueError('Resume scheduler configuration differs from checkpoint')
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state)


def build_scheduler(optimizer, config, *, epochs, steps_per_epoch):
    options = dict(config)
    name = options.pop('name')
    interval = options.pop('interval', 'epoch')
    monitor = options.pop('monitor', 'val_loss')
    if name == 'none':
        scheduler = None
    elif name == 'warmup_cosine':
        warmup = options.pop('warmup_steps', 0)
        total = options.pop('total_steps', epochs * steps_per_epoch)
        if total < 1 or not 0 <= warmup < total:
            raise ValueError('warmup_steps must be below positive total_steps')
        def multiplier(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    else:
        constructors = dict(step=torch.optim.lr_scheduler.StepLR, multistep=torch.optim.lr_scheduler.MultiStepLR,
            cosine=torch.optim.lr_scheduler.CosineAnnealingLR, exponential=torch.optim.lr_scheduler.ExponentialLR,
            plateau=torch.optim.lr_scheduler.ReduceLROnPlateau, onecycle=torch.optim.lr_scheduler.OneCycleLR)
        if name == 'onecycle':
            options.setdefault('total_steps', epochs * steps_per_epoch)
        elif name == 'cosine':
            options.setdefault('T_max', epochs)
        scheduler = constructors[name](optimizer, **options)
    return Schedule(scheduler, interval, monitor)


@dataclass
class EarlyStopper:
    direction: str = 'min'
    patience: int = 20
    min_delta: float = 0.
    best: float | None = None
    wait: int = 0

    def update(self, value):
        if not math.isfinite(value):
            raise ValueError('Validation monitor must be finite')
        improved = self.best is None or ((value < self.best - self.min_delta) if self.direction == 'min'
                                         else (value > self.best + self.min_delta))
        if improved:
            self.best, self.wait = value, 0
        else:
            self.wait += 1
        return improved

    @property
    def stopped(self):
        return self.patience > 0 and self.wait >= self.patience

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)


class MetricTracker:
    def __init__(self, directory, logger, *, tensorboard=False):
        self.logger = logger
        self.stream = (directory / 'metrics.jsonl').open('a', encoding='utf-8')
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(directory / 'tensorboard'))

    def record(self, state, metrics):
        record = dict(stage=state.stage, epoch=state.epoch, batches_seen=state.batches_seen,
                      optimizer_steps=state.optimizer_steps, **metrics)
        self.stream.write(json.dumps(record, allow_nan=False) + '\n')
        self.stream.flush()
        self.logger.info('%s', record)
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, state.optimizer_steps)

    def close(self):
        self.stream.close()
        if self.writer is not None:
            self.writer.close()
