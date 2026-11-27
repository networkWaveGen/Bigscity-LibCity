"""Lazy model declarations for the new experiment runtime."""
from dataclasses import dataclass
from importlib import import_module


def resolve(path):
    module, name = path.rsplit(':', 1)
    return getattr(import_module(module), name)


@dataclass(frozen=True)
class ModelSpec:
    factory: str
    task: str = 'traffic_state_pred'
    step: str | None = None
    train: str | None = None
    requires_checkpoint: bool = True
    supports_resume: bool = True


MODELS = {
    'GRU': ModelSpec('libcity.model.traffic_speed_prediction.runtime_rnn:GRU'),
}


def register_model(name, spec):
    if name in MODELS:
        raise ValueError(f'Model {name} is already registered')
    MODELS[name] = spec


def model_spec(name):
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f'Model {name!r} is not migrated; available: {sorted(MODELS)}') from None
