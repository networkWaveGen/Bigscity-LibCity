"""Small JSON configuration loader. Layers never mutate each other."""
from copy import deepcopy
from importlib.resources import files
import json
import math
from pathlib import Path
import warnings

from libcity.model.runtime_registry import model_spec


SELECTORS = {'optimizer', 'scheduler', 'scaler', 'loss'}
PATHS = (('data', 'data_dir'), ('data', 'cache_dir'), ('runtime', 'output_dir'))


def merge_config(low, high):
    result = deepcopy(low)
    for key, value in high.items():
        previous = result.get(key)
        if isinstance(value, dict) and isinstance(previous, dict):
            if key in SELECTORS and value.get('name', previous.get('name')) != previous.get('name'):
                result[key] = deepcopy(value)
            else:
                result[key] = merge_config(previous, value)
        else:
            result[key] = deepcopy(value)
    return result


def _object(value, location):
    if not isinstance(value, dict):
        raise TypeError(f'{location} must be a JSON object')
    return value


def _paths(layer, base):
    result = deepcopy(layer)
    for block, key in PATHS:
        if block in result and isinstance(result[block], dict) and key in result[block]:
            value = result[block][key]
            if not isinstance(value, str) or not value:
                raise TypeError(f'{block}.{key} must be a nonempty path string')
            result[block][key] = str((base / Path(value).expanduser()).resolve())
    return result


def _known(value, names, path):
    _object(value, path)
    unknown = value.keys() - set(names)
    if unknown:
        raise ValueError(f'Unknown configuration keys in {path}: {sorted(unknown)}')


def _positive(value, path, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f'{path} must be an integer >= {0 if zero else 1}')


def _number(value, path, minimum=0., strict=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= minimum if strict else value < minimum):
        raise ValueError(f'{path} must be a finite number {">" if strict else ">="} {minimum}')


def validate(config, preset):
    _known(config, ('data', 'model', 'training', 'evaluation', 'runtime'), 'config')
    for block in config:
        _object(config[block], block)
    defaults = _resource('tasks', config['runtime']['task'])
    for block in config:
        _known(config[block], defaults[block], block)
    data, model, training, evaluation, runtime = (config[name] for name in
        ('data', 'model', 'training', 'evaluation', 'runtime'))
    if not isinstance(data['dataset'], str) or not data['dataset'] or Path(data['dataset']).name != data['dataset']:
        raise ValueError('data.dataset is required and must be a single dataset directory name')
    spec = model_spec(model['name'])
    if runtime['task'] != spec.task:
        raise ValueError(f'{model["name"]} belongs to task {spec.task}, not {runtime["task"]}')
    _known(model['params'], preset.get('model', {}).get('params', {}), 'model.params')
    for key, value in model['params'].items():
        default = preset['model']['params'][key]
        if isinstance(default, bool) and type(value) is not bool:
            raise TypeError(f'model.params.{key} must be a boolean')
        if type(default) is int and type(value) is not int:
            raise TypeError(f'model.params.{key} must be an integer')
        if type(default) is float and (type(value) not in (int, float) or not math.isfinite(value)):
            raise TypeError(f'model.params.{key} must be a finite number')
    _known(data['split'], ('train', 'val'), 'data.split')
    rates = [data['split'].get(name) for name in ('train', 'val')]
    if any(type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate < 1 for rate in rates) or sum(rates) >= 1:
        raise ValueError('data.split requires positive train and val rates whose sum is below 1')
    _known(data['features'], ('load_external', 'time_in_day', 'day_in_week'), 'data.features')
    _known(data['scaler'], ('name',), 'data.scaler')
    if data['scaler'].get('name') not in ('none', 'normal', 'standard', 'minmax01', 'minmax11', 'log'):
        raise ValueError('Unknown data.scaler.name')
    for key in ('input_window', 'output_window'):
        _positive(data[key], f'data.{key}')
    for block, keys in ((training, ('epochs', 'batch_size', 'accumulation_steps')), (evaluation, ('batch_size',))):
        for key in keys:
            _positive(block[key], key)
    for block in (training, evaluation):
        _positive(block['num_workers'], 'num_workers', zero=True)
    _positive(runtime['seed'], 'runtime.seed', zero=True)
    _positive(runtime['num_threads'], 'runtime.num_threads')
    for value, name in [(data['cache'], 'data.cache'), (data['pad_training'], 'data.pad_training'),
                        (training['shuffle'], 'training.shuffle'), (training['amp'], 'training.amp'),
                        (evaluation['save_predictions'], 'evaluation.save_predictions'),
                        (runtime['tensorboard'], 'runtime.tensorboard'), (runtime['deterministic'], 'runtime.deterministic')]:
        if type(value) is not bool:
            raise TypeError(f'{name} must be a JSON boolean')
    for name, value in data['features'].items():
        if type(value) is not bool:
            raise TypeError(f'data.features.{name} must be a JSON boolean')
    if training['direction'] not in ('min', 'max'):
        raise ValueError('training.direction must be min or max')
    if training['amp_dtype'] not in ('float16', 'bfloat16'):
        raise ValueError('training.amp_dtype must be float16 or bfloat16')
    if evaluation['mode'] not in ('single', 'average'):
        raise ValueError('evaluation.mode must be single or average')
    if not isinstance(evaluation['metrics'], list) or not evaluation['metrics']:
        raise ValueError('evaluation.metrics must be a nonempty list')
    allowed_metrics = {'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'EVAR'}
    if any(not isinstance(name, str) or name.removeprefix('masked_') not in allowed_metrics for name in evaluation['metrics']):
        raise ValueError('Unknown evaluation metric')
    for key, names in {
        'optimizer': ('name', 'lr', 'weight_decay', 'betas', 'eps', 'momentum', 'alpha', 'groups'),
        'scheduler': ('name', 'interval', 'step_size', 'gamma', 'milestones', 'T_max', 'eta_min', 'patience',
                      'factor', 'monitor', 'mode', 'max_lr', 'total_steps', 'pct_start', 'warmup_steps'),
        'loss': ('name', 'mask_value'), 'early_stopping': ('patience', 'min_delta')}.items():
        _known(training[key], names, f'training.{key}')
    if training['optimizer'].get('name') not in ('adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'):
        raise ValueError('Unknown optimizer name')
    optimizer = training['optimizer']
    for key in ('lr', 'weight_decay', 'eps', 'momentum', 'alpha'):
        if key in optimizer:
            _number(optimizer[key], f'optimizer.{key}', strict=key in ('lr', 'eps'))
    if not isinstance(optimizer.get('groups', []), list):
        raise TypeError('optimizer.groups must be a list')
    for group in optimizer.get('groups', []):
        _known(group, ('modules', 'lr', 'weight_decay', 'freeze'), 'optimizer.groups')
        modules = group.get('modules')
        if not (isinstance(modules, str) and modules or isinstance(modules, list) and modules
                and all(isinstance(name, str) and name for name in modules)):
            raise ValueError('optimizer group requires nonempty modules')
        if type(group.get('freeze', False)) is not bool:
            raise TypeError('optimizer group freeze must be boolean')
        for key in ('lr', 'weight_decay'):
            if key in group:
                _number(group[key], f'optimizer group {key}', strict=key == 'lr')
    scheduler = training['scheduler']
    if scheduler.get('name') not in ('none', 'step', 'multistep', 'cosine', 'exponential', 'plateau', 'onecycle', 'warmup_cosine'):
        raise ValueError('Unknown scheduler name')
    interval = scheduler.get('interval', 'epoch')
    if interval not in ('step', 'epoch', 'metric'):
        raise ValueError('scheduler.interval must be step, epoch or metric')
    if scheduler['name'] != 'none' and 'interval' not in scheduler:
        raise ValueError('scheduler.interval must be explicitly declared')
    if scheduler['name'] == 'plateau' and interval != 'metric':
        raise ValueError('plateau scheduler requires interval=metric')
    if scheduler['name'] in ('onecycle', 'warmup_cosine') and interval != 'step':
        raise ValueError('onecycle/warmup_cosine scheduler requires interval=step')
    if training['loss'].get('name') not in ('mae', 'mse', 'masked_mae', 'masked_mse'):
        raise ValueError('Unknown training.loss.name')
    _positive(training['early_stopping']['patience'], 'early_stopping.patience', zero=True)
    _number(training['early_stopping']['min_delta'], 'early_stopping.min_delta')
    if training['clip_grad_norm'] is not None:
        _number(training['clip_grad_norm'], 'training.clip_grad_norm', strict=True)
    if not isinstance(training['monitor'], str) or not training['monitor']:
        raise ValueError('training.monitor must be a nonempty string')
    for section in (training['loss'], evaluation):
        value = section.get('mask_value')
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError('mask_value must be a finite number or null')
    _object(training['strategy'], 'training.strategy')
    if runtime['exp_id'] is not None and (not isinstance(runtime['exp_id'], str) or not runtime['exp_id']
            or runtime['exp_id'] in ('.', '..') or any(c in runtime['exp_id'] for c in '/\\')):
        raise ValueError('runtime.exp_id must be a single nonempty directory name')
    if not isinstance(runtime['device'], str):
        raise TypeError('runtime.device must be a device string')


def _resource(group, name):
    path = files('libcity.config').joinpath('runtime', group, str(name) + '.json')
    if not path.is_file():
        raise ValueError(f'No runtime {group} preset for {name!r}')
    return json.loads(path.read_text(encoding='utf-8'))


def load_config(config_file=None, overrides=None, *, config=None):
    if config_file is not None and config is not None:
        raise ValueError('Supply config_file or config, not both')
    base = Path.cwd()
    if config_file is not None:
        path = Path(config_file).expanduser().resolve()
        user = _object(json.loads(path.read_text(encoding='utf-8')), 'config')
        base = path.parent
    else:
        user = _object(deepcopy(config or {}), 'config')
    cli = _object(deepcopy(overrides or {}), 'overrides')
    selectors = merge_config(user, cli)
    _known(selectors, ('data', 'model', 'training', 'evaluation', 'runtime'), 'config')
    for block in selectors:
        _object(selectors[block], block)
    name = selectors.get('model', {}).get('name')
    if not isinstance(name, str):
        raise ValueError('model.name is required')
    model_spec(name)
    task = selectors.get('runtime', {}).get('task', 'traffic_state_pred')
    defaults = _paths(_resource('tasks', task), Path.cwd())
    preset = _resource('models', name)
    protected = [('data', 'split'), ('data', 'input_window'), ('data', 'output_window'),
                 ('evaluation', 'metrics'), ('evaluation', 'mask_value'), ('evaluation', 'mode')]
    for block, key in protected:
        if key in preset.get(block, {}) and preset[block][key] != defaults[block][key]:
            raise ValueError(f'Model preset cannot silently change benchmark field {block}.{key}')
    result = merge_config(merge_config(defaults, preset), _paths(user, base))
    result = merge_config(result, _paths(cli, Path.cwd()))
    validate(result, preset)
    deviations = [f'{block}.{key}' for block, key in protected if result[block][key] != defaults[block][key]]
    if deviations:
        warnings.warn('Explicit benchmark protocol changes: ' + ', '.join(deviations), UserWarning, stacklevel=2)
    return result
