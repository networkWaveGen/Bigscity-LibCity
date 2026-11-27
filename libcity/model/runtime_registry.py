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
    prepare_resume: str | None = None
    updates_per_epoch: str | None = None


MODELS = {
    'GRU': ModelSpec('libcity.model.traffic_speed_prediction.runtime_rnn:GRU'),
    'STGCN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_stgcn:STGCN'),
    'DCRNN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_dcrnn:DCRNN',
                       step='libcity.model.traffic_speed_prediction.runtime_dcrnn:curriculum_step'),
    'Trafformer': ModelSpec('libcity.model.traffic_speed_prediction.runtime_trafformer:Trafformer'),
    'ASTGNN': ModelSpec('libcity.model.traffic_flow_prediction.runtime_astgnn:ASTGNN',
        train='libcity.executor.astgnn_train:train',prepare_resume='libcity.executor.astgnn_train:prepare_resume'),
    'MTGNN': ModelSpec('libcity.model.traffic_speed_prediction.runtime_mtgnn:MTGNN',
        train='libcity.executor.mtgnn_train:train',updates_per_epoch='libcity.executor.mtgnn_train:updates_per_epoch'),
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
