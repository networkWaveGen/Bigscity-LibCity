"""Atomic rectangular grids with explicit IDs, causal windows and shared metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from libcity.data.bundle import DataBundle, DatasetMetadata
from libcity.data.timeline import boundaries, partition_windows
from libcity.data.traffic_bundle import fit_scaler
from libcity.data.traffic_cache import prepare_arrays


class GridWindows(Dataset):
    def __init__(self, arrays, starts, offsets, horizon, output_dim, scaler, raw_mask=False):
        self.arrays, self.target_starts, self.offsets = arrays, starts, offsets
        self.horizon, self.output_dim, self.scaler = horizon, output_dim, scaler
        self.raw_mask = raw_mask

    def __len__(self):
        return len(self.target_starts)

    def __getitem__(self, index):
        start = self.target_starts[index]
        source = self.arrays['history'][start+self.offsets].copy()
        mask = np.isfinite(source)&(source!=0) if self.raw_mask else None
        # Missing historical measurements are zero-imputed only on the input;
        # original targets remain untouched for finite/zero masking.
        source = np.nan_to_num(source, nan=0.)
        source[..., :self.output_dim] = self.scaler.transform(source[..., :self.output_dim])
        raw = self.arrays['history'][start:start+self.horizon, :, :self.output_dim].copy()
        sample = {name:torch.tensor(value, dtype=torch.float32) for name,value in dict(X=source,
            history_features=self.arrays['external'][start+self.offsets].copy(),
            future_features=self.arrays['known'][start:start+self.horizon].copy(),
            y=self.scaler.transform(raw), y_raw=raw).items()}
        if mask is not None:
            sample['history_mask'] = torch.from_numpy(mask)
        return sample


def prepare_grid(data, training, evaluation, seed, logger=None):
    folder = Path(data['data_dir'])/data['dataset']
    raw = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    raw = {**raw, **raw.get('info', {})}
    geo_path = folder/(raw.get('geo_file',data['dataset'])+'.geo')
    geo = pd.read_csv(geo_path)
    if not {'geo_id','row_id','column_id'} <= set(geo):
        raise ValueError('Grid .geo requires geo_id, row_id and column_id')
    ids = geo['geo_id'].tolist()
    if not ids or geo['geo_id'].isna().any() or len(set(ids)) != len(ids):
        raise ValueError('Grid node IDs must be nonempty, distinct and finite')
    if pd.api.types.is_numeric_dtype(geo['geo_id']) and not np.isfinite(geo['geo_id']).all():
        raise ValueError('Grid node IDs must be finite')
    positions = geo[['row_id','column_id']].to_numpy(dtype=np.float64)
    if not np.isfinite(positions).all() or np.any(positions < 0) or np.any(positions != np.floor(positions)):
        raise ValueError('Grid row/column coordinates must be nonnegative finite integers')
    positions = positions.astype(np.int64)
    height,width = (int(value)+1 for value in positions.max(0))
    linear = positions[:,0]*width+positions[:,1]
    if height*width != len(ids) or len(set(linear)) != len(ids):
        raise ValueError('Grid .geo must cover a complete rectangle without duplicate cells')
    cell_index = {tuple(value):index for index,value in enumerate(positions)}
    id_index = {value:index for index,value in enumerate(ids)}
    names = raw.get('data_files',data['dataset'])
    names = [names] if isinstance(names,str) else names
    if not isinstance(names,list) or not names or any(not isinstance(name,str) or not name for name in names):
        raise ValueError('Grid data_files must contain nonempty file basenames')
    paths = [folder/(name+('.grid' if (folder/(name+'.grid')).is_file() else '.dyna')) for name in names]
    ext_path = folder/(raw.get('ext_file',data['dataset'])+'.ext')
    sources = [folder/'config.json',geo_path,*paths]+([ext_path] if data['features']['load_external'] else [])
    columns = raw.get('data_col')
    columns = [columns] if isinstance(columns,str) else columns
    if not columns:
        reserved = {'dyna_id','type','time','entity_id','row_id','column_id'}
        columns = [name for name in pd.read_csv(paths[0],nrows=0) if name not in reserved]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError('Grid data_col must select distinct numeric observation columns')
    output_dim = raw.get('output_dim',len(columns))
    interval = raw.get('time_intervals',300)
    if type(output_dim) is not int or not 1 <= output_dim <= len(columns):
        raise ValueError('Grid output_dim must select one or more leading observation columns')
    if type(interval) is not int or interval < 1 or (data['sampling']['name']=='grid_cpt' and 86400 % interval):
        raise ValueError('Grid time_intervals must be positive integer seconds; CPT requires dividing a day')
    options = data['features']
    requested = options['known_future_columns']
    ext_names = []
    if options['load_external']:
        header = pd.read_csv(ext_path,nrows=0)
        if 'time' not in header:
            raise ValueError('Grid external data require a time column')
        ext_names = raw.get('ext_col') or [name for name in header if name not in ('ext_id','time')]
        ext_names = [ext_names] if isinstance(ext_names,str) else ext_names
        if len(set(ext_names))!=len(ext_names) or not set(ext_names)<=set(header):
            raise ValueError('Grid ext_col must select unique covariates')
    if set(requested)-set(ext_names):
        raise ValueError('known_future_columns must name loaded .ext covariates')
    calendar_width = int(options['time_in_day'])+7*int(options['day_in_week'])
    known_columns = list(range(calendar_width))+[calendar_width+ext_names.index(name) for name in ext_names if name in requested]

    def build():
        ext = None
        if options['load_external']:
            ext = pd.read_csv(ext_path)
            ext.index = pd.to_datetime(ext['time'],utc=True,errors='raise')
            if ext.index.has_duplicates or ext.index.isna().any():
                raise ValueError('Grid external timestamps must be unique and finite')
        histories,externals,clocks,intervals = [],[],[],[]
        offset = 0
        for path in paths:
            frame = pd.read_csv(path)
            stamp = pd.to_datetime(frame['time'],utc=True,errors='raise')
            if stamp.isna().any():
                raise ValueError('Grid timestamps must be finite')
            times = pd.DatetimeIndex(stamp.unique()).sort_values()
            time_index = times.get_indexer(stamp)
            try:
                if {'row_id','column_id'} <= set(frame):
                    nodes = np.array([cell_index[(row,column)] for row,column in zip(frame['row_id'],frame['column_id'])])
                else:
                    nodes = np.array([id_index[value] for value in frame['entity_id']])
            except KeyError as error:
                raise ValueError('Grid observation cell/ID is absent from .geo') from error
            pairs = time_index*len(ids)+nodes
            if len(pairs) != len(times)*len(ids) or len(np.unique(pairs)) != len(pairs):
                raise ValueError('Grid observations need exactly one row per timestamp and cell')
            values = frame[columns].to_numpy(dtype=np.float64)
            if np.isinf(values).any():
                raise ValueError('Grid observations must not contain infinities')
            history = np.empty((len(times),len(ids),len(columns)),dtype=np.float64)
            history[time_index,nodes] = values
            features = []
            if options['time_in_day']:
                seconds = times.hour*3600+times.minute*60+times.second
                features.append(np.asarray(seconds,dtype=np.float64)[:,None]/86400)
            if options['day_in_week']:
                features.append(np.eye(7,dtype=np.float64)[times.day_of_week])
            if ext is not None:
                values = ext.reindex(times)[ext_names].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError('Grid external observations must cover every timestamp with finite values')
                features.append(values)
            external = np.concatenate(features,axis=1) if features else np.empty((len(times),0))
            histories.append(history)
            externals.append(external)
            clocks.append(times.to_numpy(dtype='datetime64[ns]'))
            intervals.append((offset,offset+len(times)))
            offset += len(times)
        clock = np.concatenate(clocks)
        if np.any(np.diff(clock) <= np.timedelta64(0,'s')):
            raise ValueError('Grid data_files must be chronological without overlapping timestamps')
        external = np.concatenate(externals)
        return dict(history=np.concatenate(histories),external=external,known=external[:,known_columns],
                    clock=clock,intervals=np.asarray(intervals,dtype=np.int64))

    fields = ('history','external','known','clock','intervals')
    def validate(arrays):
        if arrays['clock'].ndim!=1 or arrays['clock'].dtype.kind!='M':
            raise ValueError('Grid cached clock must be a datetime vector')
        length = len(arrays['clock'])
        if any(arrays[name].dtype.kind not in 'fiu' for name in ('history','external','known')):
            raise ValueError('Grid cached observations and covariates must be numeric')
        if (arrays['history'].shape != (length,len(ids),len(columns)) or
            arrays['external'].shape != (length,calendar_width+len(ext_names)) or
            arrays['known'].shape != (length,len(known_columns)) or np.isnat(arrays['clock']).any() or
            np.any(np.diff(arrays['clock'])<=np.timedelta64(0,'s')) or np.isinf(arrays['history']).any() or
            not np.isfinite(arrays['external']).all() or not np.isfinite(arrays['known']).all()):
            raise ValueError('Grid array dimensions do not match atomic metadata')
        spans = arrays['intervals']
        if (spans.dtype.kind not in 'iu' or spans.ndim!=2 or spans.shape[1]!=2 or np.any(spans[:,1]<=spans[:,0]) or
                not np.array_equal(spans.ravel(),np.r_[0,np.repeat(spans[1:,0],2),length])):
            raise ValueError('Grid cached file intervals must partition the timeline')
    arrays,cache = prepare_arrays(data,raw,build,logger,namespace='grid-cpt-v1',source_paths=sources,fields=fields,validate=validate)
    clock = arrays['clock']
    sampling = data['sampling']
    if sampling['name']=='grid_window':
        offsets = list(range(-data['input_window'],0))
    elif sampling['name']=='grid_cpt':
        offsets = [-index for index in range(1,sampling.get('closeness',0)+1)]
        offsets += [-index*86400//interval*sampling.get('interval_period',1) for index in range(1,sampling.get('period',0)+1)]
        offsets += [-index*7*86400//interval*sampling.get('interval_trend',1) for index in range(1,sampling.get('trend',0)+1)]
    else:
        raise ValueError('Grid data require grid_cpt or grid_window sampling')
    offsets = np.asarray(offsets,dtype=np.int64)
    horizon = data['output_window']
    starts = np.concatenate([np.arange(left-offsets.min(),right-horizon+1) for left,right in arrays['intervals']])
    gap = np.r_[False,np.diff(clock) != np.timedelta64(interval,'s')].cumsum()
    starts = starts[gap[starts+horizon-1] == gap[starts+offsets.min()]]
    cuts = boundaries(len(clock),data['split'],clock)
    selected = partition_windows(starts,horizon,cuts,len(clock))
    scaler = fit_scaler(data['scaler']['name'],arrays['history'][:cuts[0],:,:output_dim])
    layout = dict(kind=sampling['name'],length=len(offsets),sampling=dict(sampling),
        segment_order=['window'] if sampling['name']=='grid_window' else ['closeness','period','trend'],
        grid_shape=[height,width],grid_order=np.argsort(linear),node_positions=linear,
        historical_external_dim=calendar_width+len(ext_names),future_external_dim=len(known_columns),
        future_feature_names=(['time_in_day'] if options['time_in_day'] else [])+
            ([f'day_of_week_{index}' for index in range(7)] if options['day_in_week'] else [])+
            [name for name in ext_names if name in requested],
        target_semantics='raw_and_scaled',split_semantics='target_time_v1')
    if sampling['name']=='grid_window':
        layout['history_mask'] = 'finite_nonzero_raw_observations'
    metadata = DatasetMetadata(data['dataset'],len(ids),len(columns),output_dim,
        info=dict(node_ids=ids,data_columns=columns,time_intervals=interval),input_layout=layout)
    generator = torch.Generator().manual_seed(seed)
    loaders,intervals = [],{}
    for index,(name,indices) in enumerate(zip(('train','val','test'),selected)):
        chosen = starts[indices]
        if not len(chosen):
            raise ValueError(f'{name} has no complete grid target windows; increase timeline or adjust sampling')
        intervals[name] = dict(samples=len(chosen),target_index_start=int(chosen.min()),target_index_end=int(chosen.max()+horizon-1))
        if index==0 and data['pad_training']:
            chosen = np.pad(chosen,(0,(-len(chosen))%training['batch_size']),mode='edge')
        dataset = GridWindows(arrays,chosen,offsets,horizon,output_dim,scaler,raw_mask=sampling['name']=='grid_window')
        cfg = training if index==0 else evaluation
        loaders.append(DataLoader(dataset,batch_size=cfg['batch_size'],num_workers=cfg['num_workers'],
            shuffle=training['shuffle'] if index==0 else False,generator=generator if index==0 else None))
    summary = dict(semantics='target_time_v1',adapter=sampling['name'],grid_shape=[height,width],input_window=len(offsets),
        history_span_points=int(-offsets.min()),requested_input_window=data['input_window'],offsets=offsets.tolist(),
        intervals=intervals,train=len(loaders[0].dataset),val=len(loaders[1].dataset),test=len(loaders[2].dataset),
        train_end=str(clock[cuts[0]]),val_end=str(clock[cuts[1]]),scaler_fit_end_exclusive=str(clock[cuts[0]]),
        train_padded=data['pad_training'],cache=cache,external_scaling='none',input_missing_observations='zero')
    if logger:
        logger.info('Grid target-time split: %s',summary)
    return DataBundle(*loaders,metadata,scaler,generator,summary)
