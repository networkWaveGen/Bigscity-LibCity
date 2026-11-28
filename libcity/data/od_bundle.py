"""Dense OD atoms preserve origin/destination IDs, target time and raw masks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset

from libcity.data.bundle import DataBundle,DatasetMetadata
from libcity.data.timeline import boundaries,partition_windows
from libcity.data.traffic_bundle import fit_scaler
from libcity.data.traffic_cache import prepare_arrays


class ODWindows(Dataset):
    def __init__(self,arrays,starts,history,horizon,outputs,scaler):
        self.arrays,self.target_starts = arrays,np.asarray(starts,dtype=np.int64)
        self.offsets,self.horizon,self.output_dim,self.scaler = np.arange(-history,0),horizon,outputs,scaler

    def __len__(self):
        return len(self.target_starts)

    def __getitem__(self,index):
        start = self.target_starts[index]
        indices = start+self.offsets
        observed = self.arrays['history'][indices].copy()
        mask = np.isfinite(observed)&(observed>0)
        source = np.nan_to_num(observed,nan=0.)
        source[..., :self.output_dim] = self.scaler.transform(source[..., :self.output_dim])
        raw = self.arrays['history'][start:start+self.horizon,...,:self.output_dim].copy()
        sample = {name:torch.tensor(value,dtype=torch.float32) for name,value in dict(
            X=source,y=self.scaler.transform(raw),y_raw=raw).items()}
        sample['history_mask'] = torch.from_numpy(mask)
        if self.arrays['external'].shape[-1]:
            sample['history_features'] = torch.tensor(self.arrays['external'][indices],dtype=torch.float32)
        if self.arrays['known'].shape[-1]:
            sample['future_features'] = torch.tensor(self.arrays['known'][start:start+self.horizon],dtype=torch.float32)
        return sample


def prepare_od(data,training,evaluation,seed,logger=None):
    if data['sampling']['name']!='od_window':
        raise ValueError('OD data require data.sampling.name=od_window')
    if data['graph']['name'] not in ('weighted','binary','none'):
        raise ValueError('OD graph must use raw weighted/binary relations or explicit none')
    if data['graph']['name']=='none' and any(data['graph'].get(key,False) for key in ('symmetric','self_loops')):
        raise ValueError('OD graph none cannot enable symmetric or self_loops')
    folder = Path(data['data_dir'])/data['dataset']
    raw = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    raw = {**raw,**raw.get('info',{})}
    geo_path = folder/(raw.get('geo_file',data['dataset'])+'.geo')
    geo = pd.read_csv(geo_path)
    if 'geo_id' not in geo or geo['geo_id'].isna().any() or geo['geo_id'].duplicated().any() or not len(geo):
        raise ValueError('OD .geo requires nonempty unique node IDs')
    if pd.api.types.is_numeric_dtype(geo['geo_id']) and not np.isfinite(geo['geo_id']).all():
        raise ValueError('OD node IDs must be finite')
    ids = geo['geo_id'].tolist()
    node_index = {node:index for index,node in enumerate(ids)}
    positions,shape,cell_index = None,None,{}
    if {'row_id','column_id'}<=set(geo):
        positions = geo[['row_id','column_id']].to_numpy(dtype=np.float64)
        if not np.isfinite(positions).all() or np.any(positions<0) or np.any(positions!=np.floor(positions)):
            raise ValueError('OD grid positions must be finite nonnegative integers')
        positions = positions.astype(np.int64)
        shape = [int(value)+1 for value in positions.max(0)]
        cell_index = {tuple(value):index for index,value in enumerate(positions)}
        if len(cell_index)!=len(ids) or shape[0]*shape[1]!=len(ids):
            raise ValueError('OD .geo grid must cover a complete rectangle')
    names = raw.get('data_files',data['dataset'])
    names = [names] if isinstance(names,str) else names
    if not isinstance(names,list) or not names or any(not isinstance(name,str) or not name for name in names):
        raise ValueError('OD data_files must contain nonempty file basenames')
    paths = [next((folder/(name+suffix) for suffix in ('.od','.gridod','.dyna')
        if (folder/(name+suffix)).is_file()),folder/(name+'.od')) for name in names]
    columns = raw.get('data_col')
    columns = [columns] if isinstance(columns,str) else columns
    reserved = {'dyna_id','type','time','origin_id','destination_id','origin_row_id','origin_column_id',
        'destination_row_id','destination_column_id'}
    columns = columns or [column for column in pd.read_csv(paths[0],nrows=0) if column not in reserved]
    if not columns or len(set(columns))!=len(columns):
        raise ValueError('OD data_col must select distinct observation columns')
    outputs,interval = raw.get('output_dim',len(columns)),raw.get('time_intervals',300)
    if type(outputs) is not int or not 1<=outputs<=len(columns):
        raise ValueError('OD output_dim must select leading observation columns')
    if type(interval) is not int or interval<1:
        raise ValueError('OD time_intervals must be positive integer seconds')
    ext_path = folder/(raw.get('ext_file',data['dataset'])+'.ext')
    rel_path = folder/(raw.get('rel_file',data['dataset'])+'.rel')
    options = data['features']
    ext_names = []
    if options['load_external']:
        ext_names = raw.get('ext_col')
        ext_names = [ext_names] if isinstance(ext_names,str) else ext_names
        ext_names = ext_names or [column for column in pd.read_csv(ext_path,nrows=0) if column not in ('ext_id','time')]
        if len(set(ext_names))!=len(ext_names):
            raise ValueError('OD ext_col must select unique covariates')
    requested = options['known_future_columns']
    if set(requested)-set(ext_names):
        raise ValueError('OD known_future_columns must name loaded .ext covariates')
    calendar_names = (['time_of_day'] if options['time_in_day'] else [])+(
        [f'day_of_week_{index}' for index in range(7)] if options['day_in_week'] else [])
    known_indices = list(range(len(calendar_names)))+[len(calendar_names)+ext_names.index(name) for name in requested]
    sources = [folder/'config.json',geo_path,*paths]+([rel_path] if data['graph']['name']!='none' else [])+(
        [ext_path] if options['load_external'] else [])

    def build():
        ext = None
        if options['load_external']:
            ext = pd.read_csv(ext_path)
            ext.index = pd.to_datetime(ext['time'],utc=True,errors='raise')
            if ext.index.has_duplicates or ext.index.isna().any():
                raise ValueError('OD external timestamps must be unique and finite')
        histories,externals,clocks,segments = [],[],[],[]
        offset = 0
        for path in paths:
            frame = pd.read_csv(path)
            if 'time' not in frame or not set(columns)<=set(frame):
                raise ValueError('OD files require time and the selected observation columns')
            stamp = pd.to_datetime(frame['time'],utc=True,errors='raise')
            if stamp.isna().any():
                raise ValueError('OD timestamps must be finite')
            times = pd.DatetimeIndex(stamp.unique()).sort_values()
            if not len(times):
                raise ValueError('OD files must contain at least one timestamp')
            time_index = times.get_indexer(stamp)
            try:
                if {'origin_id','destination_id'}<=set(frame):
                    origin = np.array([node_index[value] for value in frame['origin_id']])
                    destination = np.array([node_index[value] for value in frame['destination_id']])
                elif cell_index:
                    origin = np.array([cell_index[(row,column)] for row,column in zip(frame['origin_row_id'],frame['origin_column_id'])])
                    destination = np.array([cell_index[(row,column)] for row,column in zip(frame['destination_row_id'],frame['destination_column_id'])])
                else:
                    raise ValueError('OD observations require origin_id/destination_id or complete grid coordinates')
            except KeyError as error:
                raise ValueError('OD origin/destination ID or grid cell is absent from .geo') from error
            pairs = (time_index*len(ids)+origin)*len(ids)+destination
            if len(pairs)!=len(times)*len(ids)**2 or len(np.unique(pairs))!=len(pairs):
                raise ValueError('OD data need exactly one row per timestamp, origin and destination; encode zero flows explicitly')
            values = frame[columns].to_numpy(dtype=np.float64)
            if np.isinf(values).any():
                raise ValueError('OD observations must not contain infinities')
            history = np.empty((len(times),len(ids),len(ids),len(columns)),dtype=np.float64)
            history[time_index,origin,destination] = values
            features = []
            if options['time_in_day']:
                features.append(np.asarray(times.hour*3600+times.minute*60+times.second,dtype=np.float64)[:,None]/86400)
            if options['day_in_week']:
                features.append(np.eye(7,dtype=np.float64)[times.day_of_week])
            if ext is not None:
                covariates = ext.reindex(times)[ext_names].to_numpy(dtype=np.float64)
                if not np.isfinite(covariates).all():
                    raise ValueError('OD external data must cover every timestamp with finite values')
                features.append(covariates)
            external = np.concatenate(features,1) if features else np.empty((len(times),0))
            histories.append(history);externals.append(external)
            clocks.append(times.to_numpy(dtype='datetime64[ns]'))
            segments.append((offset,offset+len(times)));offset += len(times)
        clock = np.concatenate(clocks)
        if np.any(np.diff(clock)<=np.timedelta64(0,'s')):
            raise ValueError('OD data_files must be chronological without overlapping timestamps')
        graph = np.zeros((len(ids),len(ids)),dtype=np.float32)
        if data['graph']['name']!='none' and rel_path.is_file():
            relation = pd.read_csv(rel_path)
            weight = raw.get('weight_col') or next((name for name in relation if name not in ('rel_id','type','origin_id','destination_id')),None)
            if len(relation) and not isinstance(weight,str) and data['graph']['name']=='weighted':
                raise ValueError('Weighted OD relations require one weight_col')
            used = set()
            for _,row in relation.iterrows():
                try:
                    pair = node_index[row['origin_id']],node_index[row['destination_id']]
                except KeyError as error:
                    raise ValueError('OD relation ID is absent from .geo') from error
                if pair in used:
                    raise ValueError('OD relations require distinct origin/destination pairs')
                used.add(pair)
                value = float(row[weight]) if data['graph']['name']=='weighted' else 1.
                if not np.isfinite(value) or value<0:
                    raise ValueError('OD relation weights must be finite and nonnegative')
                graph[pair] = value
        if data['graph'].get('symmetric',False):
            graph = np.maximum(graph,graph.T)
        if data['graph'].get('self_loops',False):
            np.fill_diagonal(graph,1.)
        external = np.concatenate(externals)
        return dict(history=np.concatenate(histories),external=external,known=external[:,known_indices],
            clock=clock,intervals=np.asarray(segments,dtype=np.int64),graph=graph)

    fields = ('history','external','known','clock','intervals','graph')
    def validate(arrays):
        length = len(arrays['clock'])
        if any(arrays[name].dtype.kind not in 'fiu' for name in ('history','external','known','graph')):
            raise ValueError('OD cached observations and graph must be numeric')
        if (arrays['history'].shape!=(length,len(ids),len(ids),len(columns)) or
                arrays['external'].shape!=(length,len(calendar_names)+len(ext_names)) or
                arrays['known'].shape!=(length,len(known_indices)) or arrays['graph'].shape!=(len(ids),len(ids)) or
                arrays['clock'].dtype.kind!='M' or np.isnat(arrays['clock']).any() or
                np.any(np.diff(arrays['clock'])<=np.timedelta64(0,'s')) or np.isinf(arrays['history']).any() or
                not np.isfinite(arrays['external']).all() or not np.isfinite(arrays['known']).all() or
                not np.isfinite(arrays['graph']).all() or np.any(arrays['graph']<0)):
            raise ValueError('OD cached arrays do not match atomic metadata')
        if arrays['intervals'].dtype.kind not in 'iu' or arrays['intervals'].ndim!=2 or arrays['intervals'].shape[1]!=2 or np.any(
                arrays['intervals'][:,1]<=arrays['intervals'][:,0]) or not np.array_equal(
                arrays['intervals'].ravel(),np.r_[0,np.repeat(arrays['intervals'][1:,0],2),length]):
            raise ValueError('OD cached file intervals must partition the timeline')
    arrays,cache = prepare_arrays(data,raw,build,logger,namespace='od-window-v1',source_paths=sources,fields=fields,validate=validate)
    clock = arrays['clock']
    cuts = boundaries(len(clock),data['split'],clock)
    scaler = fit_scaler(data['scaler']['name'],arrays['history'][:cuts[0],...,:outputs])
    history,horizon = data['input_window'],data['output_window']
    starts = np.concatenate([np.arange(left+history,right-horizon+1) for left,right in arrays['intervals']])
    gaps = np.r_[False,np.diff(clock)!=np.timedelta64(interval,'s')].cumsum()
    starts = starts[gaps[starts-history]==gaps[starts+horizon-1]]
    selections = partition_windows(starts,horizon,cuts,len(clock))
    layout = dict(kind='od_window',length=history,spatial_axes=['origin','destination'],node_ids=ids,
        grid_shape=shape,grid_positions=positions.tolist() if positions is not None else None,
        history_features=dict(kind='global_covariates',columns=calendar_names+ext_names),
        future_feature_names=[(calendar_names+ext_names)[index] for index in known_indices],
        history_mask='positive_finite_raw_observations',target_semantics='raw_and_scaled',split_semantics='target_time_v1')
    metadata = DatasetMetadata(data['dataset'],len(ids),len(columns),outputs,arrays['graph'],
        dict(raw,node_ids=ids,data_columns=columns,time_intervals=interval),layout)
    generator = torch.Generator().manual_seed(seed)
    loaders,splits = [],{}
    for index,(name,selection) in enumerate(zip(('train','val','test'),selections)):
        chosen = starts[selection]
        if not len(chosen):
            raise ValueError(f'{name} has no complete OD target windows; increase timeline or adjust windows/split')
        splits[name] = dict(samples=len(chosen),target_index_start=int(chosen.min()),target_index_end=int(chosen.max()+horizon-1))
        if index==0 and data['pad_training']:
            chosen = np.pad(chosen,(0,(-len(chosen))%training['batch_size']),mode='edge')
        dataset = ODWindows(arrays,chosen,history,horizon,outputs,scaler)
        config = training if index==0 else evaluation
        loaders.append(DataLoader(dataset,batch_size=config['batch_size'],num_workers=config['num_workers'],
            shuffle=training['shuffle'] if index==0 else False,generator=generator if index==0 else None))
    summary = dict(semantics='target_time_v1',adapter='od_window',input_window=history,requested_input_window=history,
        spatial_shape=[len(ids),len(ids)],grid_shape=shape,intervals=splits,train=len(loaders[0].dataset),
        val=len(loaders[1].dataset),test=len(loaders[2].dataset),train_padded=data['pad_training'],
        train_end=str(clock[cuts[0]]),val_end=str(clock[cuts[1]]),scaler_fit_end_exclusive=str(clock[cuts[0]]),
        discarded_boundary_windows=len(starts)-sum(map(len,selections)),cache=cache,
        sampling_gap_policy='file_and_full_input_target_span_contiguous',input_missing_observations='zero')
    if logger:
        logger.info('OD target-time split: %s',summary)
    return DataBundle(*loaders,metadata,scaler,generator,summary)
