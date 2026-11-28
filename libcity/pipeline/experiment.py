"""Shared Python/CLI experiment entry; task assembly stays out of scripts."""
import json
import logging
from pathlib import Path
import platform
import random
import uuid

import numpy as np
import torch

from libcity.config.runtime import load_config, _resource
from libcity.data.bundle import prepare_traffic
from libcity.executor.evaluate import ExperimentResult, ForecastLoss, evaluate_loop
from libcity.executor.train import build_context, train_loop
from libcity.executor.algorithm import AlgorithmContext
from libcity.model.runtime_registry import model_spec, resolve
from libcity.utils.runtime_checkpoint import CheckpointManager,inference_spec


def select_device(value, *, allowed=('cpu','cuda','mps')):
    if value == 'auto':
        value = ('cuda:0' if 'cuda' in allowed and torch.cuda.is_available() else
                 'mps' if 'mps' in allowed and torch.backends.mps.is_available() else
                 'cpu' if 'cpu' in allowed else None)
        if value is None:
            raise ValueError(f'No available device supported by this model: {allowed}')
    device = torch.device(value)
    if device.type == 'cuda' and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f'Requested CUDA device is unavailable: {value}')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Requested MPS device is unavailable')
    if device.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError(f'Unsupported device {value}')
    return device


def initialize_runtime(runtime):
    seed = runtime['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(runtime['num_threads'])
    torch.backends.cudnn.deterministic = runtime['deterministic']
    torch.backends.cudnn.benchmark = not runtime['deterministic']
    torch.use_deterministic_algorithms(runtime['deterministic'], warn_only=True)
    return select_device(runtime['device'])


def run_logger(directory):
    logger = logging.getLogger('libcity.run.' + uuid.uuid4().hex)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in (logging.FileHandler(directory / 'run.log', encoding='utf-8'), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def protocol_changes(config):
    defaults = _resource('tasks', config['runtime']['task'])
    fields = [('data', 'split'), ('data', 'input_window'), ('data', 'output_window'), ('data','sampling'),
              ('evaluation', 'metrics'), ('evaluation', 'mask_value'), ('evaluation', 'mode')]
    changes = {f'{block}.{key}': config[block][key] for block, key in fields
            if key in defaults[block] and key in config[block] and config[block][key] != defaults[block][key]}
    protocol = model_spec(config['model']['name']).protocol
    if protocol:
        changes['benchmark.protocol'] = protocol
    return changes


def run_experiment(mode, config_file=None, *, config=None, overrides=None, checkpoint=None, resume=None, evaluate_test=True):
    if mode not in ('train', 'evaluate'):
        raise ValueError('Experiment mode must be train or evaluate')
    cfg = load_config(config_file, overrides, config=config)
    spec = model_spec(cfg['model']['name'])
    if type(evaluate_test) is not bool or (not evaluate_test and (mode!='train' or not spec.supports_validation)):
        raise ValueError('Validation-only fitting requires a trainable validation-capable model')
    if mode == 'train' and not spec.requires_checkpoint and spec.train is None:
        raise ValueError('This algorithm provides direct evaluation only')
    if resume is not None and (mode != 'train' or not spec.supports_resume):
        raise ValueError('This experiment does not support --resume')
    if mode == 'train' and checkpoint is not None:
        raise ValueError('Use --resume last.pt for training, or --checkpoint best.pt for evaluation')
    if mode == 'evaluate' and spec.requires_checkpoint and checkpoint is None:
        raise ValueError('Evaluation requires --checkpoint (best.pt or fitted algorithm weights)')
    if checkpoint is not None and not spec.requires_checkpoint:
        raise ValueError('This algorithm does not use a checkpoint')
    for label, path in (('checkpoint', checkpoint), ('resume', resume)):
        if path is not None:
            from libcity.utils.runtime_checkpoint import require_pt
            require_pt(path)
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f'{label} file does not exist: {path}')
    if cfg['runtime']['device']=='auto':
        cfg['runtime']['device']=str(select_device('auto',allowed=spec.device_types))
    device = initialize_runtime(cfg['runtime'])
    if device.type not in spec.device_types:
        raise ValueError(f'{cfg["model"]["name"]} supports devices {spec.device_types}')
    cfg['runtime']['device'] = str(device)
    cfg['runtime']['exp_id'] = cfg['runtime']['exp_id'] or uuid.uuid4().hex[:12]
    directory = Path(cfg['runtime']['output_dir']) / cfg['runtime']['exp_id']
    if directory.exists() and mode == 'train' and resume is None:
        raise FileExistsError(f'Experiment directory exists: {directory}; choose another exp_id or resume')
    directory.mkdir(parents=True, exist_ok=True)
    logger = run_logger(directory)
    context = None
    fitted = None
    try:
        if spec.prepare_data:
            adapter=resolve(spec.prepare_data)
        elif cfg['runtime']['task']=='traffic_state_pred':
            adapter=prepare_traffic
        elif cfg['runtime']['task']=='map_matching':
            from libcity.data.map_matching_bundle import prepare_map_matching
            adapter=prepare_map_matching
        else:
            raise ValueError('Task adapter is not migrated yet')
        bundle = adapter(cfg['data'], cfg['training'], cfg['evaluation'], cfg['runtime']['seed'], logger=logger)
        arguments=dict(metadata=bundle.metadata,**cfg['model']['params'])
        if spec.task=='traffic_state_pred':
            arguments.update(input_window=cfg['data']['input_window'],output_window=cfg['data']['output_window'])
        model = resolve(spec.factory)(**arguments)
        if spec.kind=='neural':
            model=model.to(device)
        changes = protocol_changes(cfg)
        if (spec.task=='traffic_state_pred' and bundle.split_summary['input_window']!=cfg['data']['input_window']):
            changes['data.effective_input_window'] = bundle.split_summary['input_window']
        if changes:
            logger.warning('Explicit benchmark protocol changes: %s', changes)
        (directory / 'resolved_config.json').write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
        info = dict(python=platform.python_version(), pytorch=str(torch.__version__), numpy=np.__version__,
            platform=platform.platform(), mode=mode, execution_kind=spec.kind, device=str(device), seed=cfg['runtime']['seed'],
            benchmark_protocol=spec.protocol or 'task_default',
            split=bundle.split_summary, protocol_changes=changes,
            capabilities=dict(requires_checkpoint=spec.requires_checkpoint, supports_resume=spec.supports_resume,
                              supports_validation=spec.supports_validation))
        (directory / 'runtime_info.json').write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
        if mode=='train' and spec.kind=='external':
            fitted = resolve(spec.train)(AlgorithmContext(model,bundle,cfg,device,directory,logger))
            if not isinstance(fitted,ExperimentResult):
                raise ValueError('External training must return an ExperimentResult')
        elif mode == 'train':
            updates = resolve(spec.updates_per_epoch)(cfg,bundle) if spec.updates_per_epoch else None
            loss_fn=resolve(spec.loss)(bundle,cfg['training']['loss']) if spec.loss else None
            evaluator=resolve(spec.evaluate_loader) if spec.evaluate_loader else None
            context = build_context(model, bundle, cfg, device, directory, logger,steps_per_epoch=updates,
                loss_fn=loss_fn,evaluate_fn=evaluator)
            if resume is not None:
                prepare = resolve(spec.prepare_resume) if spec.prepare_resume else None
                context.checkpoints.load_last(resume, context,prepare=prepare)
            if spec.train is not None:
                score = resolve(spec.train)(context)
            else:
                step = resolve(spec.step)(context.loss_fn) if spec.step else None
                score = train_loop(context, step)
            if not isinstance(score, (float, int)) or not np.isfinite(score):
                raise ValueError('Training entry must return a finite scalar validation score')
        elif spec.requires_checkpoint:
            CheckpointManager.load_weights(checkpoint, model,
                input_spec=inference_spec(bundle,cfg) if spec.kind=='neural' else None)
        if mode=='train' and not evaluate_test:
            result=ExperimentResult({'validation_score':float(score)})
        elif spec.evaluate:
            result=resolve(spec.evaluate)(model,bundle,cfg['evaluation'],directory=directory/'eval',logger=logger)
        else:
            evaluator=resolve(spec.evaluate_loader) if spec.evaluate_loader else evaluate_loop
            loss_fn=resolve(spec.loss)(bundle,cfg['training']['loss']) if spec.loss else ForecastLoss(cfg['training']['loss'],bundle.scaler,bundle.metadata.output_dim)
            result = evaluator(model, bundle.test, device=device,
                loss_fn=loss_fn,
                evaluation=cfg['evaluation'], directory=directory / 'eval')
        result.artifacts.update(resolved_config=directory / 'resolved_config.json', runtime_info=directory / 'runtime_info.json',
                                run_log=directory / 'run.log')
        if context is not None:
            result.metrics['best_validation'] = float(score)
            result.artifacts.update(best=directory / 'checkpoints/best.pt', last=directory / 'checkpoints/last.pt')
        if fitted is not None:
            result.metrics.update(fitted.metrics)
            result.artifacts.update(fitted.artifacts)
        logger.info('Result metrics: %s', result.metrics)
        return result
    finally:
        if context is not None:
            context.tracker.close()
        close_logger(logger)
