"""Small JSON configuration loader. Layers never mutate each other."""
from copy import deepcopy
from importlib.resources import files
import json
import math
from pathlib import Path
import warnings

from libcity.model.runtime_registry import model_spec


SELECTORS = {'optimizer', 'scheduler', 'scaler', 'loss', 'graph', 'sampling'}
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


def _validate_runtime(runtime):
    _positive(runtime['seed'],'runtime.seed',zero=True)
    _positive(runtime['num_threads'],'runtime.num_threads')
    for key in ('tensorboard','deterministic'):
        if type(runtime[key]) is not bool:
            raise TypeError(f'runtime.{key} must be boolean')
    if runtime['exp_id'] is not None and (not isinstance(runtime['exp_id'],str) or not runtime['exp_id']
            or runtime['exp_id'] in ('.','..') or any(c in runtime['exp_id'] for c in '/\\')):
        raise ValueError('runtime.exp_id must be a single nonempty directory name')
    if not isinstance(runtime['device'],str):
        raise TypeError('runtime.device must be a device string')


def validate(config, preset):
    _known(config, ('data', 'model', 'training', 'evaluation', 'runtime'), 'config')
    for block in config:
        _object(config[block], block)
    defaults = task_defaults(config['runtime']['task'], config['model']['name'])
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
    _validate_runtime(runtime)
    if spec.kind=='external':
        _validate_external(data,training,evaluation,runtime,model['name'])
        return
    if spec.task=='map_matching':
        if type(data['cache']) is not bool:
            raise TypeError('data.cache must be boolean')
        if data['cache']:
            raise ValueError('The map-matching adapter currently requires data.cache=false')
        if type(evaluation['save_predictions']) is not bool:
            raise TypeError('evaluation.save_predictions must be boolean')
        if (not isinstance(evaluation['metrics'],list) or not evaluation['metrics']
                or any(name not in ('RMF','AN','AL') for name in evaluation['metrics'])):
            raise ValueError('Map-matching metrics must be a nonempty list of RMF/AN/AL')
        return
    _known(data['split'], ('train', 'val', 'train_end', 'val_end'), 'data.split')
    rates = [data['split'].get(name) for name in ('train', 'val')]
    if any(type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate < 1 for rate in rates) or sum(rates) >= 1:
        raise ValueError('data.split requires positive train and val rates whose sum is below 1')
    if spec.task=='traffic_state_pred':
        if ('train_end' in data['split']) != ('val_end' in data['split']):
            raise ValueError('Explicit splitting requires train_end and val_end')
        for key in ('train_end','val_end'):
            if key in data['split'] and not isinstance(data['split'][key],str):
                raise TypeError(f'data.split.{key} must be an ISO timestamp')
        _known(data['features'], ('load_external', 'time_in_day', 'day_in_week', 'known_future_columns'), 'data.features')
        _known(data['scaler'], ('name',), 'data.scaler')
        graph_keys = {'HGCN':('cluster_nodes','cluster_labels','cluster_epsilon'),
            'GTS':('history_channel','history_steps')}.get(model['name'],())
        _known(data['graph'], ('name', 'symmetric', 'self_loops', 'epsilon','reachability_column','distance_column')+graph_keys, 'data.graph')
        if model['name']=='HGCN':
            for key,value in dict(cluster_nodes=0,cluster_labels=None,cluster_epsilon=.1).items():
                data['graph'].setdefault(key,value)
            _positive(data['graph']['cluster_nodes'],'graph.cluster_nodes',zero=True)
            _number(data['graph']['cluster_epsilon'],'graph.cluster_epsilon')
            if data['graph']['cluster_epsilon']>1:
                raise ValueError('graph.cluster_epsilon must be between zero and one')
            labels = data['graph']['cluster_labels']
            if labels is not None and (not isinstance(labels,list) or not labels or any(type(value) is not int or value<0 for value in labels)):
                raise ValueError('graph.cluster_labels must be a nonempty list of nonnegative integers or null')
        if model['name']=='GTS':
            for key in ('history_channel','history_steps'):
                data['graph'].setdefault(key,0)
                _positive(data['graph'][key],'graph.'+key,zero=True)
        if 'distance_column' in data['graph']:
            column = data['graph']['distance_column']
            if model['name']!='MultiSTGCnet' or (column is not None and (not isinstance(column,str) or not column)):
                raise ValueError('data.graph.distance_column is a MultiSTGCnet column name or null')
        if 'reachability_column' in data['graph']:
            column = data['graph']['reachability_column']
            if model['name']!='TGCLSTM' or (column is not None and (not isinstance(column,str) or not column)):
                raise ValueError('data.graph.reachability_column is a TGCLSTM column name or null')
        _known(data['sampling'], ('name','closeness','period','trend','interval_period','interval_trend',
                                  'temporal_history','period_steps','trend_steps'), 'data.sampling')
        if data['sampling'].get('name') not in ('window','cpt','crann','grid_cpt','frame_cpt','seasonal_windows'):
            raise ValueError('Unknown data.sampling.name')
        if (spec.protocol=='grid_cpt') != (data['sampling']['name']=='grid_cpt'):
            raise ValueError('grid_cpt sampling requires a registered grid_cpt model and vice versa')
        if (spec.protocol in ('convgcn_windows','reslstm_windows')) != (data['sampling']['name']=='seasonal_windows'):
            raise ValueError('seasonal_windows sampling requires CONVGCN/ResLSTM and vice versa')
        if (model['name']=='MultiSTGCnet') != (data['sampling']['name']=='frame_cpt'):
            raise ValueError('frame_cpt sampling requires MultiSTGCnet and vice versa')
        if data['sampling']['name']=='seasonal_windows':
            for key in ('period_steps','trend_steps'):
                _positive(data['sampling'].get(key),f'data.sampling.{key}')
        if data['sampling']['name']=='crann':
            _positive(data['sampling'].get('temporal_history'), 'data.sampling.temporal_history')
        if data['sampling']['name'] in ('cpt','grid_cpt','frame_cpt'):
            for key in ('closeness','period','trend'):
                _positive(data['sampling'].get(key,0),f'data.sampling.{key}',zero=True)
            if not sum(data['sampling'].get(key,0) for key in ('closeness','period','trend')):
                raise ValueError('CPT sampling requires at least one history segment')
            for key in ('interval_period','interval_trend'):
                _positive(data['sampling'].get(key,1),f'data.sampling.{key}')
        if data['graph'].get('name') not in ('binary', 'weighted', 'distance','inverse_distance','distance_bands'):
            raise ValueError('Unknown data.graph.name')
        if (data['graph']['name']=='distance_bands' and model['name']!='MultiSTGCnet') or (
                model['name']=='MultiSTGCnet' and data['graph']['name'] not in ('distance_bands','binary')):
            raise ValueError('MultiSTGCnet requires raw distance_bands or explicit unit-distance binary graph')
        for key in ('symmetric', 'self_loops'):
            if type(data['graph'].get(key, False)) is not bool:
                raise TypeError(f'data.graph.{key} must be boolean')
        if data['scaler'].get('name') not in ('none', 'normal', 'standard', 'minmax01', 'minmax11', 'log'):
            raise ValueError('Unknown data.scaler.name')
        for key in ('input_window', 'output_window'):
            _positive(data[key], f'data.{key}')
    for block, keys in ((training, ('epochs', 'batch_size', 'accumulation_steps')), (evaluation, ('batch_size',))):
        for key in keys:
            _positive(block[key], key)
    for block in (training, evaluation):
        _positive(block['num_workers'], 'num_workers', zero=True)
    boolean_values = [(data['cache'],'data.cache'), (training['shuffle'],'training.shuffle'),
                      (training['amp'],'training.amp'), (evaluation['save_predictions'],'evaluation.save_predictions')]
    if spec.task=='traffic_state_pred':
        boolean_values.append((data['pad_training'],'data.pad_training'))
        for name,value in data['features'].items():
            if name=='known_future_columns':
                if not isinstance(value,list) or any(not isinstance(item,str) for item in value) or len(set(value))!=len(value):
                    raise TypeError('known_future_columns must be a list of unique column names')
            else:
                boolean_values.append((value,'data.features.'+name))
    for value,name in boolean_values:
        if type(value) is not bool:
            raise TypeError(f'{name} must be a JSON boolean')
    if training['direction'] not in ('min', 'max'):
        raise ValueError('training.direction must be min or max')
    if training['amp_dtype'] not in ('float16', 'bfloat16'):
        raise ValueError('training.amp_dtype must be float16 or bfloat16')
    if spec.task=='traffic_state_pred' and evaluation['mode'] not in ('single', 'average'):
        raise ValueError('evaluation.mode must be single or average')
    if not isinstance(evaluation['metrics'], list) or not evaluation['metrics']:
        raise ValueError('evaluation.metrics must be a nonempty list')
    allowed_metrics = {'MAE', 'MSE', 'RMSE', 'MAPE', 'R2', 'EVAR'}
    if spec.task=='traj_loc_pred':
        if not isinstance(evaluation['topk'],list) or not evaluation['topk'] or any(type(k) is not int or k<1 for k in evaluation['topk']):
            raise ValueError('evaluation.topk requires positive integers')
        allowed_metrics = {f'{name}@{k}' for name in ('Recall','MRR','NDCG') for k in evaluation['topk']}
    elif spec.task=='road_representation':
        allowed_metrics={'loss','MAE','MSE','RMSE'} if spec.protocol=='road_reconstruction' else {'loss','accuracy'}
    elif spec.task=='eta':
        allowed_metrics={'MAE','MSE','RMSE','MAPE'}
    if any(not isinstance(name, str) or name.removeprefix('masked_') not in allowed_metrics for name in evaluation['metrics']):
        raise ValueError('Unknown evaluation metric')
    for key, names in {
        'optimizer': ('name', 'lr', 'weight_decay', 'betas', 'eps', 'momentum', 'alpha', 'groups'),
        'scheduler': ('name', 'interval', 'step_size', 'gamma', 'milestones', 'T_max', 'eta_min', 'patience',
                      'factor', 'monitor', 'mode', 'max_lr', 'total_steps', 'pct_start', 'warmup_steps',
                      'model_dim', 'lr_multiplier'),
        'loss': ('name', 'mask_value','alpha','eps')+(('delta',) if training['loss'].get('name') in ('huber','masked_huber') else ())+(('negative_samples',) if spec.protocol in ('atst_ranking','stan_ranking') else
            ('negative_samples','temperature','num_nearest','exclude_visited') if spec.protocol=='geosan_ranking' else ()),
        'early_stopping': ('patience', 'min_delta')}.items():
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
    if scheduler.get('name') not in ('none', 'step', 'multistep', 'cosine', 'exponential', 'plateau', 'onecycle', 'warmup_cosine', 'inverse_sqrt_cosine'):
        raise ValueError('Unknown scheduler name')
    interval = scheduler.get('interval', 'epoch')
    if interval not in ('step', 'epoch', 'metric'):
        raise ValueError('scheduler.interval must be step, epoch or metric')
    if scheduler['name'] != 'none' and 'interval' not in scheduler:
        raise ValueError('scheduler.interval must be explicitly declared')
    if scheduler['name'] == 'plateau' and interval != 'metric':
        raise ValueError('plateau scheduler requires interval=metric')
    if scheduler['name'] in ('onecycle', 'warmup_cosine', 'inverse_sqrt_cosine') and interval != 'step':
        raise ValueError('onecycle/warmup_cosine/inverse_sqrt_cosine scheduler requires interval=step')
    if scheduler['name']=='inverse_sqrt_cosine':
        _positive(scheduler.get('model_dim'),'scheduler.model_dim')
        _positive(scheduler.get('warmup_steps'),'scheduler.warmup_steps')
        _number(scheduler.get('lr_multiplier',1.),'scheduler.lr_multiplier',strict=True)
    road_losses = ('mae','mse') if spec.protocol=='road_reconstruction' else ('negative_sampling',)
    trajectory_losses = {'atst_ranking':('normalized_bpr',),'stan_ranking':('sampled_softmax',),
        'cara_ranking':('bpr',),'geosan_ranking':('weighted_binary',)}.get(spec.protocol,('nll',))
    eta_losses = ('cumulative_relative_rmse',) if spec.protocol=='ttpnet_eta' else ('mae','mse')
    losses={'traj_loc_pred':trajectory_losses,'road_representation':road_losses,'eta':eta_losses}.get(
        spec.task,('mae','mse','masked_mae','masked_mse','huber','masked_huber'))
    if training['loss'].get('name') not in losses:
        raise ValueError('Unknown training.loss.name')
    if training['loss']['name'] in ('huber','masked_huber'):
        _number(training['loss'].get('delta',1.),'training.loss.delta',strict=True)
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
    if model['name']=='GTS':
        strategy = training['strategy']
        _known(strategy,('use_curriculum_learning','cl_decay_steps','temperature','graph_regularization_epochs',
            'graph_regularization_weight'),'training.strategy')
        if type(strategy.get('use_curriculum_learning')) is not bool:
            raise ValueError('GTS strategy.use_curriculum_learning must be boolean')
        _positive(strategy.get('cl_decay_steps'),'strategy.cl_decay_steps')
        _positive(strategy.get('graph_regularization_epochs'),'strategy.graph_regularization_epochs',zero=True)
        _number(strategy.get('temperature'),'strategy.temperature',strict=True)
        _number(strategy.get('graph_regularization_weight'),'strategy.graph_regularization_weight')
    if spec.task=='traj_loc_pred':
        _number(data['session_gap_hours'],'session_gap_hours',strict=True)
        _positive(data['min_session_length'],'min_session_length')
        if spec.protocol in ('atst_ranking','stan_ranking','geosan_ranking'):
            _positive(training['loss'].get('negative_samples'),'loss.negative_samples')
        if spec.protocol=='geosan_ranking':
            _positive(data['quadkey_level'],'data.quadkey_level')
            _positive(data['quadkey_ngram'],'data.quadkey_ngram')
            if not data['quadkey_ngram']<=data['quadkey_level']<=23:
                raise ValueError('GeoSAN requires quadkey_ngram <= quadkey_level <= 23')
            _positive(training['loss']['num_nearest'],'loss.num_nearest')
            _number(training['loss']['temperature'],'loss.temperature',strict=True)
            if type(training['loss']['exclude_visited']) is not bool:
                raise TypeError('GeoSAN loss.exclude_visited must be boolean')
    if spec.task=='road_representation':
        _positive(data['negative_samples'],'negative_samples',zero=True)
        if spec.protocol=='road_reconstruction':
            for key in ('feature_columns','categorical_columns'):
                if (not isinstance(data[key],list) or any(not isinstance(value,str) or not value for value in data[key])
                        or len(set(data[key]))!=len(data[key])):
                    raise ValueError(f'data.{key} must contain unique nonempty column names')
            _known(data['scaler'], ('name',), 'data.scaler')
            if data['scaler'].get('name') not in ('none','standard'):
                raise ValueError('Road reconstruction supports none/standard scaling')
        elif training['loss']['name']!='negative_sampling':
            raise ValueError('LINE training requires loss.name=negative_sampling')
    if spec.task=='eta':
        if training['loss']['name'] not in eta_losses:
            raise ValueError(f'ETA training requires one of {eta_losses}')
        if spec.protocol=='ttpnet_eta':
            _positive(data['min_route_length'],'data.min_route_length')
            if data['min_route_length']<2:
                raise ValueError('TTPNet min_route_length must be at least two')
        if not 0<=training['loss'].get('alpha',.3)<=1 or training['loss'].get('eps',10.)<=0:
            raise ValueError('ETA requires alpha in [0,1] and eps>0')
    if spec.task!='traffic_state_pred' and data['cache']:
        raise ValueError('This task adapter currently requires data.cache=false')


def _validate_external(data,training,evaluation,runtime,model):
    if type(data['cache']) is not bool or data['cache']:
        raise ValueError('Walk graph adapter requires data.cache=false')
    _known(data['graph'],('name','directed','self_loops','weight_column'),'data.graph')
    # A selector change replaces the old block; fill the selected graph's own defaults.
    for name,value in dict(directed=True,self_loops=False,weight_column=None).items():
        data['graph'].setdefault(name,value)
    if data['graph']['name'] not in ('binary','weighted'):
        raise ValueError('Walk graph.name must be binary or weighted')
    if model=='DeepWalk' and data['graph']['name']!='binary':
        raise ValueError('DeepWalk uses unweighted binary edges')
    for name in ('directed','self_loops'):
        if type(data['graph'][name]) is not bool: raise TypeError(f'Walk graph.{name} must be boolean')
    column = data['graph']['weight_column']
    if column is not None and (not isinstance(column,str) or not column): raise ValueError('Walk weight_column must be a column name or null')
    for name in ('epochs','num_workers','window_size','batch_words'): _positive(training[name],f'training.{name}')
    _number(training['subsample'],'training.subsample')
    if type(training['dynamic_window']) is not bool: raise TypeError('Walk dynamic_window must be boolean')
    _known(training['optimizer'],('name','lr','min_lr'),'training.optimizer')
    if training['optimizer']['name']!='gensim_sgd': raise ValueError('Walk training uses gensim_sgd, not a PyTorch optimizer')
    for name in ('lr','min_lr'): _number(training['optimizer'][name],f'optimizer.{name}',strict=True)
    if training['optimizer']['min_lr']>training['optimizer']['lr']: raise ValueError('Walk min_lr must not exceed lr')
    _known(training['loss'],('name','negative_samples','hierarchical_softmax'),'training.loss')
    if training['loss']['name']!='word2vec': raise ValueError('Walk training requires word2vec loss')
    _positive(training['loss']['negative_samples'],'loss.negative_samples',zero=True)
    if type(training['loss']['hierarchical_softmax']) is not bool: raise TypeError('hierarchical_softmax must be boolean')
    if not training['loss']['hierarchical_softmax'] and not training['loss']['negative_samples']: raise ValueError('Word2Vec needs hierarchical softmax or negative samples')
    if runtime['deterministic'] and training['num_workers']!=1: raise ValueError('Deterministic Word2Vec requires training.num_workers=1')
    if runtime['tensorboard']: raise ValueError('External Word2Vec records JSONL; runtime.tensorboard must be false')
    if type(evaluation['save_predictions']) is not bool: raise TypeError('save_predictions must be boolean')
    if not isinstance(evaluation['metrics'],list) or not evaluation['metrics'] or any(name not in ('nodes','dimensions','coverage','mean_norm','isolated_nodes','dead_end_nodes') for name in evaluation['metrics']):
        raise ValueError('Walk metrics are structural diagnostics, not held-out quality scores')


def _resource(group, name):
    path = files('libcity.config').joinpath('runtime', group, str(name) + '.json')
    if not path.is_file():
        raise ValueError(f'No runtime {group} preset for {name!r}')
    return json.loads(path.read_text(encoding='utf-8'))


def task_defaults(task, model):
    """Named task protocols keep distinct objectives out of model hyperparameters."""
    spec = model_spec(model)
    if spec.kind=='external':
        return _resource('protocols',spec.protocol)
    defaults = _resource('tasks', task)
    protocol = spec.protocol
    return merge_config(defaults, _resource('protocols', protocol)) if protocol else defaults


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
    task = selectors.get('runtime', {}).get('task', model_spec(name).task)
    defaults = _paths(task_defaults(task, name), Path.cwd())
    defaults['model']['name']=name
    user_model=user.get('model',{})
    if isinstance(user_model,dict) and user_model.get('name',name)!=name:
        # A CLI model switch selects the new preset rather than retaining the
        # previous model's architecture fields from the user JSON.
        user['model']={'name':name}
    preset = _resource('models', name)
    protected = [('data', 'split'), ('data', 'input_window'), ('data', 'output_window'),
                 ('evaluation', 'metrics'), ('evaluation', 'mask_value'), ('evaluation', 'mode')]
    for block, key in protected:
        if key in preset.get(block, {}) and preset[block][key] != defaults[block].get(key):
            raise ValueError(f'Model preset cannot silently change benchmark field {block}.{key}')
    result = merge_config(merge_config(defaults, preset), _paths(user, base))
    result = merge_config(result, _paths(cli, Path.cwd()))
    validate(result, preset)
    deviations = [f'{block}.{key}' for block, key in protected
                  if key in defaults[block] and result[block][key] != defaults[block][key]]
    if deviations:
        warnings.warn('Explicit benchmark protocol changes: ' + ', '.join(deviations), UserWarning, stacklevel=2)
    return result
