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
    if model['name']=='GEML':
        for key in ('embed_dim','p_interval'):
            _positive(model['params'][key],'GEML '+key)
    if model['name']=='CSTN':
        for key in ('n_layers','c_lt'):
            _positive(model['params'][key],'CSTN '+key)
    if model['name']=='STDN':
        for key in ('cnn_flat_size','lstm_out_size'):
            _positive(model['params'][key],'STDN '+key)
    if model['name']=='HIEST':
        params=model['params']
        for key in ('blocks','layers','kernel_size','residual_channels','dilation_channels','skip_channels','end_channels','global_nodes'):
            _positive(params[key],'HIEST '+key)
        _positive(params['order'],'HIEST order',zero=True)
        _number(params['dropout'],'HIEST dropout')
        if params['dropout']>=1 or params['adjtype'] not in ('doubletransition','transition','symnadj','normlap','scalap','identity'):
            raise ValueError('HIEST dropout or adjtype is invalid')
        for key in ('n1','n2','n3','n4'): _number(params[key],'HIEST '+key)
    if model['name']=='ESG':
        params=model['params']
        for key in ('n_blocks','n_layers','residual_channels','end_channels','st_embedding_dim',
                    'dilation_exp','conv_channels','dy_embedding_dim','skip_channels'):
            _positive(params[key],'ESG '+key)
        _positive(params['gcn_depth'],'ESG gcn_depth',zero=True)
        kernels=params['kernel_set']
        if (not isinstance(kernels,list) or not kernels or any(type(v) is not int or v<1 for v in kernels) or
                any(b<=a for a,b in zip(kernels,kernels[1:])) or params['conv_channels']%len(kernels)):
            raise ValueError('ESG requires increasing positive kernels and conv_channels divisible by their number')
        intervals=params['dy_interval']
        if not isinstance(intervals,list) or len(intervals)!=params['n_layers'] or any(type(v) is not int or v<1 for v in intervals):
            raise ValueError('ESG dy_interval must contain one positive interval per layer')
        if not 0<=params['dropout']<1 or not 0<=params['propalpha']<=1:
            raise ValueError('ESG requires dropout in [0,1) and propalpha in [0,1]')
    if model['name']=='GSNet':
        for key in ('num_of_gru_layers','gru_hidden_size'):
            _positive(model['params'][key],'GSNet '+key)
        filters = model['params']['gcn_nums_filters']
        if not isinstance(filters,list) or not filters or any(type(value) is not int or value<1 for value in filters):
            raise ValueError('GSNet gcn_nums_filters must be a nonempty list of positive integers')
    if model['name']=='DSAN':
        params = model['params']
        for key in ('L','d','n_h','B'):
            _positive(params[key],'DSAN '+key)
        _positive(params['l_d'],'DSAN l_d',zero=True)
        _number(params['r_d'],'DSAN r_d')
        if params['d']%params['n_h'] or params['r_d']>=1:
            raise ValueError('DSAN requires d divisible by n_h and r_d below one')
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
        stream_keys = ('primary_index','auxiliary_index') if model['name']=='DMSTGCN' else ()
        risk_keys = ('graph_columns','external_scaler') if model['name']=='GSNet' else ()
        if model['name']=='STDN':
            _known(data['features'],('load_external','time_in_day','day_in_week','known_future_columns','flow_scaler'),'STDN data.features')
            if any(data['features'][key] for key in ('load_external','time_in_day','day_in_week','known_future_columns')):
                raise ValueError('STDN uses volume/flow context without external or known-future switches')
            _known(data['features']['flow_scaler'],('name',),'STDN features.flow_scaler')
            if data['features']['flow_scaler'].get('name') not in ('none','normal','standard','minmax01','minmax11','log'):
                raise ValueError('Unknown STDN flow_scaler.name')
        _known(data['features'], ('load_external', 'time_in_day', 'day_in_week', 'known_future_columns')+stream_keys+risk_keys+(('flow_scaler',) if model['name']=='STDN' else ()), 'data.features')
        if risk_keys:
            columns = data['features']['graph_columns']
            if columns is not None and (not isinstance(columns,list) or not columns or
                    any(not isinstance(value,str) or not value for value in columns) or len(set(columns))!=len(columns)):
                raise ValueError('GSNet features.graph_columns requires unique nonempty observation names or null')
            _known(data['features']['external_scaler'],('name',),'data.features.external_scaler')
            if data['features']['external_scaler'].get('name') not in ('none','normal','standard','minmax01','minmax11','log'):
                raise ValueError('Unknown GSNet external_scaler.name')
        if stream_keys:
            for key in stream_keys:
                _positive(data['features'][key],'features.'+key,zero=True)
            if data['features']['primary_index']==data['features']['auxiliary_index']:
                raise ValueError('DMSTGCN primary_index and auxiliary_index must be distinct')
            _known(data['graph'],('name','symmetric','self_loops'),'data.graph')
            if data['graph']['name']!='dynamic' or data['graph'].get('symmetric',False) or data['graph'].get('self_loops',False):
                raise ValueError('DMSTGCN requires its dynamic graph without static graph switches')
        _known(data['scaler'], ('name',), 'data.scaler')
        graph_keys = {'HGCN':('cluster_nodes','cluster_labels','cluster_epsilon'),
            'GTS':('history_channel','history_steps'),
            'CCRNN':('svd_rank','svd_channels','normalization'),
            'GSNet':('road_column','risk_column','poi_column','use_poi'),
            'ESG':('history_steps','history_channels'),
            'HIEST':('region_threshold','region_groups','reconstruction')}.get(model['name'],())
        _known(data['graph'], ('name', 'symmetric', 'self_loops', 'epsilon','reachability_column','distance_column')+graph_keys, 'data.graph')
        if model['name']=='HIEST':
            for key,value in dict(region_threshold=.8,region_groups=None,reconstruction='binary').items():
                data['graph'].setdefault(key,value)
            _number(data['graph']['region_threshold'],'HIEST graph.region_threshold')
            groups=data['graph']['region_groups']
            if groups is not None and (not isinstance(groups,list) or not groups or any(not isinstance(group,list) or not group or
                    any(type(value) not in (int,float,str) or isinstance(value,str) and not value or
                        type(value) in (int,float) and not math.isfinite(value) for value in group) for group in groups)):
                raise ValueError('HIEST region_groups requires nonempty lists of finite real node IDs or null')
            if data['graph']['reconstruction'] not in ('binary','weights'):
                raise ValueError('HIEST reconstruction must be binary or weights')
            if data['sampling']['name']!='window': raise ValueError('HIEST requires continuous window sampling')
        if model['name']=='ESG':
            _known(data['graph'],('name','symmetric','self_loops','history_steps','history_channels'),'ESG data.graph')
            if data['graph']['name']!='dynamic' or data['graph'].get('symmetric',False) or data['graph'].get('self_loops',False):
                raise ValueError('ESG requires dynamic graphs without static graph switches')
            _positive(data['graph']['history_steps'],'ESG graph.history_steps',zero=True)
            channels=data['graph']['history_channels']
            if channels is not None and (not isinstance(channels,list) or not channels or
                    any(type(value) is not int or value<0 for value in channels) or len(set(channels))!=len(channels)):
                raise ValueError('ESG history_channels requires unique nonnegative target indices or null')
        if model['name']=='GSNet':
            _known(data['graph'],('name','symmetric','self_loops')+graph_keys,'GSNet data.graph')
            if data['graph']['name']!='risk_multigraph':
                raise ValueError('GSNet requires risk_multigraph relations')
            if type(data['graph'].get('use_poi')) is not bool:
                raise TypeError('GSNet graph.use_poi must be boolean')
            for key in ('road_column','risk_column','poi_column'):
                value = data['graph'].get(key)
                if value is not None and (not isinstance(value,str) or not value):
                    raise ValueError('GSNet graph.'+key+' must be a column name or null')
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
        if model['name']=='CCRNN':
            _known(data['graph'],('name','svd_rank','svd_channels','normalization','symmetric','self_loops'),'data.graph')
            if data['graph']['name']!='svd_gaussian' or data['graph'].get('symmetric',False) or data['graph'].get('self_loops',False):
                raise ValueError('CCRNN requires svd_gaussian initialization without extra symmetric/self_loops switches')
            _positive(data['graph']['svd_rank'],'graph.svd_rank')
            channels = data['graph']['svd_channels']
            if channels is not None and (not isinstance(channels,list) or not channels or
                    any(type(value) is not int or value<0 for value in channels) or len(set(channels))!=len(channels)):
                raise ValueError('CCRNN svd_channels must be null or unique nonnegative observation indices')
            if data['graph']['normalization'] not in ('randomwalk','laplacian'):
                raise ValueError('CCRNN normalization must be randomwalk or laplacian')
        if 'distance_column' in data['graph']:
            column = data['graph']['distance_column']
            if model['name']!='MultiSTGCnet' or (column is not None and (not isinstance(column,str) or not column)):
                raise ValueError('data.graph.distance_column is a MultiSTGCnet column name or null')
        if 'reachability_column' in data['graph']:
            column = data['graph']['reachability_column']
            if model['name']!='TGCLSTM' or (column is not None and (not isinstance(column,str) or not column)):
                raise ValueError('data.graph.reachability_column is a TGCLSTM column name or null')
        stdn_keys=('att_lstm_num','att_lstm_seq_len','hist_feature_daynum','last_feature_num','cnn_nbhd_size','nbhd_size') if model['name']=='STDN' else ()
        _known(data['sampling'], ('name','closeness','period','trend','interval_period','interval_trend',
                                  'temporal_history','period_steps','trend_steps')+stdn_keys, 'data.sampling')
        if data['sampling'].get('name') not in ('window','cpt','crann','grid_cpt','frame_cpt','seasonal_windows','od_window','grid_window','risk_cpt','stdn_grid'):
            raise ValueError('Unknown data.sampling.name')
        if (spec.protocol=='stdn_grid') != (data['sampling']['name']=='stdn_grid'):
            raise ValueError('stdn_grid sampling requires STDN and vice versa')
        if model['name']=='STDN':
            _known(data['sampling'],('name',)+stdn_keys,'STDN data.sampling')
            for key in stdn_keys:
                _positive(data['sampling'][key],'STDN sampling.'+key,zero=key not in ('att_lstm_num','att_lstm_seq_len'))
            if data['sampling']['att_lstm_seq_len']%2!=1:
                raise ValueError('STDN periodic attention sequence length must be odd')
            _known(data['graph'],('name','symmetric','self_loops'),'STDN data.graph')
            if data['graph']['name']!='none' or data['graph'].get('symmetric',False) or data['graph'].get('self_loops',False):
                raise ValueError('STDN uses observed OD flows, requiring graph.name=none without graph switches')
        if (spec.protocol=='risk_cpt') != (data['sampling']['name']=='risk_cpt'):
            raise ValueError('risk_cpt sampling requires a registered risk_cpt model and vice versa')
        if spec.protocol=='risk_cpt':
            _known(data['sampling'],('name','closeness','period','trend','interval_period','interval_trend'),'GSNet data.sampling')
        if (spec.protocol=='grid_window') != (data['sampling']['name']=='grid_window'):
            raise ValueError('grid_window sampling requires a registered grid_window model and vice versa')
        if model['name']=='DSAN':
            _known(data['sampling'],('name',),'DSAN data.sampling')
            _known(data['graph'],('name','symmetric','self_loops'),'DSAN data.graph')
            if data['graph']['name']!='none' or data['graph'].get('symmetric',False) or data['graph'].get('self_loops',False):
                raise ValueError('DSAN uses grid positions, requiring graph.name=none without graph switches')
        if (spec.protocol=='od_window') != (data['sampling']['name']=='od_window'):
            raise ValueError('od_window sampling requires a registered OD model and vice versa')
        if spec.protocol=='od_window' and model['name']!='CSTN' and data['graph']['name'] not in ('weighted','binary'):
            raise ValueError('OD models require raw weighted or binary relations')
        if spec.protocol=='od_window':
            _known(data['graph'],('name','symmetric','self_loops'),'OD data.graph')
        if model['name']=='CSTN' and (data['graph']['name']!='none' or data['graph'].get('symmetric',False) or data['graph'].get('self_loops',False)):
            raise ValueError('CSTN uses grid positions, requiring graph.name=none without graph switches')
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
        if data['sampling']['name'] in ('cpt','grid_cpt','frame_cpt','risk_cpt'):
            for key in ('closeness','period','trend'):
                _positive(data['sampling'].get(key,0),f'data.sampling.{key}',zero=True)
            if not sum(data['sampling'].get(key,0) for key in ('closeness','period','trend')):
                raise ValueError('CPT sampling requires at least one history segment')
            for key in ('interval_period','interval_trend'):
                _positive(data['sampling'].get(key,1),f'data.sampling.{key}')
        if data['graph'].get('name') not in ('binary', 'weighted', 'distance','inverse_distance','distance_bands')+(
                ('svd_gaussian',) if model['name']=='CCRNN' else ('dynamic',) if model['name'] in ('DMSTGCN','ESG') else
                ('risk_multigraph',) if model['name']=='GSNet' else ('none',) if model['name'] in ('CSTN','DSAN','STDN') else ()):
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
            elif name in ('load_external','time_in_day','day_in_week'):
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
    if model['name']=='GSNet':
        ranking = {'Precision','Recall','F1-Score','MAP','PCC'}
        allowed_metrics |= ranking
        _positive(evaluation['topk'],'GSNet evaluation.topk')
        if evaluation['mask_value'] is not None:
            raise ValueError('GSNet evaluation includes real zero risks; mask_value must be null')
        if any(isinstance(name,str) and name.startswith('masked_') and name.removeprefix('masked_') in ranking for name in evaluation['metrics']):
            raise ValueError('GSNet ranking metrics use region eligibility, without masked_ prefixes')
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
                      'factor', 'monitor', 'mode', 'threshold', 'max_lr', 'total_steps', 'pct_start', 'warmup_steps',
                      'model_dim', 'lr_multiplier'),
        'loss': ('name', 'mask_value','alpha','eps')+(('original_weight','regional_weight','orthogonal_weight') if model['name']=='HIEST' else ())+(('thresholds','weights') if model['name']=='GSNet' else ())+(('od_weight','origin_weight','destination_weight') if model['name']=='GEML' else ())+(('delta',) if training['loss'].get('name') in ('huber','masked_huber') else ())+(('negative_samples',) if spec.protocol in ('atst_ranking','stan_ranking') else
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
    if 'threshold' in scheduler:
        if scheduler['name']!='plateau':
            raise ValueError('scheduler.threshold is only valid for plateau')
        _number(scheduler['threshold'],'scheduler.threshold')
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
    if model['name'] in ('CCRNN','DSAN'):
        losses += ('rmse','masked_rmse')
    if model['name']=='HIEST':
        for key,default in dict(original_weight=.01,regional_weight=.01,orthogonal_weight=.1).items():
            training['loss'].setdefault(key,default)
            _number(training['loss'][key],'HIEST loss.'+key)
    if model['name']=='GSNet':
        losses = ('risk_mse',)
        loss = training['loss']
        if loss.get('mask_value') is not None:
            raise ValueError('GSNet risk loss includes real zero risks; mask_value must be null')
        thresholds,weights = loss.get('thresholds'),loss.get('weights')
        for key,values in (('thresholds',thresholds),('weights',weights)):
            if values is not None:
                if not isinstance(values,list):
                    raise ValueError('GSNet loss.'+key+' must be a numeric list or null')
                for value in values: _number(value,'GSNet loss.'+key)
        if thresholds is not None and any(right<=left for left,right in zip(thresholds,thresholds[1:])):
            raise ValueError('GSNet loss.thresholds must be strictly increasing')
        if weights is not None and (not weights or not any(weights) or thresholds is not None and len(weights)!=len(thresholds)+1):
            raise ValueError('GSNet loss.weights must have positive mass and match the risk bins')
    if model['name']=='GEML':
        losses = ('mse','masked_mse')
        weights = [training['loss'].get(key) for key in ('od_weight','origin_weight','destination_weight')]
        for key,value in zip(('od_weight','origin_weight','destination_weight'),weights):
            _number(value,'GEML training.loss.'+key)
        if not sum(weights):
            raise ValueError('GEML requires at least one positive loss weight')
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
    if model['name']=='RGSL':
        _known(training['strategy'],('temperature',),'training.strategy')
        _number(training['strategy'].get('temperature'),'RGSL strategy.temperature',strict=True)
    if model['name']=='CCRNN':
        _known(training['strategy'],('use_curriculum_learning','cl_decay_steps'),'training.strategy')
        if type(training['strategy'].get('use_curriculum_learning')) is not bool:
            raise ValueError('CCRNN use_curriculum_learning must be boolean')
        _positive(training['strategy']['cl_decay_steps'],'strategy.cl_decay_steps')
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
