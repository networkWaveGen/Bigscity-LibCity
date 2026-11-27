"""Safe tensor-only inference weights and independent complete training snapshots."""
from dataclasses import asdict
from collections.abc import Mapping
import os
from pathlib import Path
import random

import numpy as np
import torch

from libcity.utils.checkpoint import copy_state_to_cpu


def require_pt(path):
    path=Path(path)
    if path.suffix.lower()!='.pt':
        raise ValueError('Runtime checkpoints must be .pt files: best.pt for evaluation, last.pt for resume')
    return path


def resume_settings(config):
    data = {key: value for key, value in config['data'].items() if key not in ('data_dir', 'cache_dir', 'cache')}
    training = {key: value for key, value in config['training'].items() if key != 'epochs'}
    if (config['training']['scheduler']['name'] in ('onecycle', 'warmup_cosine')
            or 'fine_tune_epochs' in config['training']['strategy']):
        training['epochs'] = config['training']['epochs']
    return dict(data=data, model=config['model'], training=training)


def plain(value):
    if isinstance(value,Mapping):
        return {key:plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [plain(item) for item in value]
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    return value


def data_state(bundle):
    metadata=bundle.metadata
    return dict(name=metadata.name,num_nodes=metadata.num_nodes,feature_dim=metadata.feature_dim,
        output_dim=metadata.output_dim,info=plain(metadata.info),input_layout=plain(metadata.input_layout),
        scaler=dict(name=type(bundle.scaler).__name__,state=plain(vars(bundle.scaler))))


def rng_state():
    numpy = np.random.get_state()
    state = dict(python=random.getstate(), numpy=dict(name=numpy[0], keys=numpy[1].tolist(),
                 position=numpy[2], has_gauss=numpy[3], cached_gaussian=numpy[4]), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, 'get_rng_state'):
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy['name'], np.asarray(numpy['keys'], dtype=np.uint32), numpy['position'],
                         numpy['has_gauss'], numpy['cached_gaussian']))
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])


class CheckpointManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.best_state = None

    def _save(self, name, payload):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        temporary = path.with_suffix('.tmp')
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def save_best(self, model):
        self.best_state = copy_state_to_cpu(model.state_dict())
        return self._save('best.pt', self.best_state)

    def save_last(self, context):
        payload = dict(format_version=2, model=copy_state_to_cpu(context.model.state_dict()),
            optimizer=copy_state_to_cpu(context.optimizer.state_dict()), scheduler=context.schedule.state_dict(),
            amp_scaler=context.amp_scaler.state_dict(), train_state=asdict(context.state),
            early_stopper=context.stopper.state_dict(), best_model=self.best_state, rng=rng_state(),
            loader_rng=context.bundle.generator.get_state() if context.bundle.generator is not None else None,
            model_name=context.config['model']['name'], model_params=context.config['model']['params'],
            resume_settings=resume_settings(context.config),data_state=data_state(context.bundle))
        return self._save('last.pt', payload)

    def load_last(self, path, context, prepare=None):
        payload = torch.load(require_pt(path), map_location='cpu', weights_only=True)
        required = {'format_version','model', 'optimizer', 'scheduler', 'train_state', 'rng',
                    'early_stopper', 'best_model', 'amp_scaler','model_name','model_params',
                    'resume_settings','data_state'}
        if not isinstance(payload, dict) or not required <= payload.keys():
            raise ValueError('Resume requires a complete last.pt; best.pt contains inference weights only')
        if payload['model_name'] != context.config['model']['name'] or payload['model_params'] != context.config['model']['params']:
            raise ValueError('Resume model configuration does not match checkpoint')
        if payload['resume_settings'] != resume_settings(context.config):
            raise ValueError('Resume data/training configuration does not match checkpoint')
        if payload.get('format_version')!=2 or payload.get('data_state')!=data_state(context.bundle):
            raise ValueError('Resume dataset metadata/normalization state does not match checkpoint')
        if prepare is not None:
            prepare(context,payload['train_state'])
        context.model.load_state_dict(payload['model'])
        context.optimizer.load_state_dict(payload['optimizer'])
        context.schedule.load_state_dict(payload['scheduler'])
        context.amp_scaler.load_state_dict(payload['amp_scaler'])
        for key, value in payload['train_state'].items():
            setattr(context.state, key, value)
        context.stopper.load_state_dict(payload['early_stopper'])
        self.best_state = payload['best_model']
        if self.best_state is not None:
            self._save('best.pt', self.best_state)
        if payload.get('loader_rng') is not None and context.bundle.generator is not None:
            context.bundle.generator.set_state(payload['loader_rng'])
        restore_rng(payload['rng'])
        # A completed/early-stopped resume may perform no further epoch, but
        # its new experiment directory must still contain the full last state.
        self._save('last.pt',payload)
        return payload

    @staticmethod
    def load_weights(path, model):
        state = torch.load(require_pt(path), map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError('Evaluation requires a pure model state_dict (best.pt)')
        model.load_state_dict(state)
