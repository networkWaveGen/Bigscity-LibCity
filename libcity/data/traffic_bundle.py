"""Atomic traffic bridge with explicit forecast inputs and target-time splitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from libcity.data.bundle import DataBundle, DatasetMetadata, METADATA_FIELDS
from libcity.data.timeline import boundaries, partition_windows
from libcity.utils.normalization import (NoneScaler, StandardScaler, NormalScaler,
                                        MinMax01Scaler, MinMax11Scaler, LogScaler)


def fit_scaler(name, values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError('Training interval has no finite observations for normalization')
    if name == 'standard':
        return StandardScaler(float(finite.mean()), float(finite.std()))
    if name == 'normal':
        return NormalScaler(float(finite.max()))
    if name in ('minmax01', 'minmax11'):
        return (MinMax01Scaler if name == 'minmax01' else MinMax11Scaler)(float(finite.min()), float(finite.max()))
    return {'none': NoneScaler, 'log': LogScaler}[name]()


class TrafficWindows(Dataset):
    def __init__(self, history, targets, known, starts, offsets, horizon, output_dim, scaler, clock=None):
        self.history, self.targets, self.known = history, targets, known
        self.clock = clock
        self.target_starts = np.asarray(starts, dtype=np.int64)
        self.offsets, self.horizon, self.output_dim, self.scaler = offsets, horizon, output_dim, scaler

    def __len__(self):
        return len(self.target_starts)

    def __getitem__(self, index):
        start = self.target_starts[index]
        source = self.history[start + self.offsets].copy()
        source[..., :self.output_dim] = self.scaler.transform(source[..., :self.output_dim])
        raw = self.targets[start:start+self.horizon].copy()
        result = dict(X=torch.tensor(source, dtype=torch.float32),
                      y=torch.tensor(self.scaler.transform(raw), dtype=torch.float32),
                      y_raw=torch.tensor(raw, dtype=torch.float32))
        if self.known.shape[-1]:
            result['future_features'] = torch.tensor(self.known[start:start+self.horizon], dtype=torch.float32)
        return result


def _read_atomic(data, metadata, logger=None):
    from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
    folder = Path(data['data_dir']) / data['dataset']
    graph_config, options = data['graph'], data['features']
    legacy = dict(metadata, dataset=data['dataset'], data_dir=data['data_dir'], cache_dir=data['cache_dir'],
        cache_dataset=False, init_weight_inf_or_zero='inf' if graph_config['name'] in ('distance','inverse_distance') else 'zero',
        set_weight_link_or_dist='link' if graph_config['name'] in ('binary','distance_bands') else 'dist',
        calculate_weight_adj=graph_config['name']=='distance', distance_inverse=graph_config['name']=='inverse_distance',
        weight_adj_epsilon=graph_config.get('epsilon', .1),
        add_time_in_day=options['time_in_day'], add_day_in_week=options['day_in_week'])
    if graph_config['name']=='distance_bands' and graph_config.get('distance_column'):
        legacy['weight_col'] = graph_config['distance_column']
    if graph_config['name']=='none':
        class ObservationReader(TrafficStatePointDataset):
            def _load_rel(self):
                self.adj_mx = np.zeros((self.num_nodes,self.num_nodes),dtype=np.float32)
        reader = ObservationReader(legacy,logger=logger)
    else:
        reader = TrafficStatePointDataset(legacy, logger=logger)
    filenames = metadata.get('data_files', data['dataset'])
    filenames = [filenames] if isinstance(filenames, str) else filenames
    histories, targets, knowns, clocks, intervals = [], [], [], [], []
    known_indices = []
    ext = reader._load_ext() if options['load_external'] else None
    ext_names = reader.ext_col if isinstance(reader.ext_col, list) else [reader.ext_col]
    if ext is not None and (ext_names == [''] or not ext_names):
        import pandas as pd
        ext_names = list(pd.read_csv(folder/(reader.ext_file+'.ext'), nrows=0).columns[2:])
    requested = options.get('known_future_columns', [])
    if set(requested) - set(ext_names if ext is not None else []):
        raise ValueError('known_future_columns must name loaded .ext covariates; observations/targets cannot be future inputs')
    offset = 0
    for filename in filenames:
        observed = reader._load_dyna(filename)
        times = reader.timesolts.copy()
        if np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, 's')):
            raise ValueError('Traffic timestamps must be finite, unique and strictly increasing')
        history = reader._add_external_information(observed, ext)
        known_indices = list(range(observed.shape[-1], observed.shape[-1] + int(options['time_in_day']) + 7*int(options['day_in_week'])))
        ext_offset = observed.shape[-1] + len(known_indices)
        known_indices += [ext_offset+i for i, name in enumerate(ext_names) if ext is not None and name in requested]
        histories.append(history)
        targets.append(observed[..., :reader.output_dim])
        knowns.append(history[..., known_indices])
        clocks.append(times)
        intervals.append((offset, offset+len(times)))
        offset += len(times)
    clock = np.concatenate(clocks)
    if np.any(np.diff(clock) <= np.timedelta64(0, 's')):
        raise ValueError('data_files must be in chronological order without overlapping timestamps')
    history, target, known = map(np.concatenate, (histories, targets, knowns))
    graph = reader.adj_mx.copy()
    if graph_config['name']=='inverse_distance' and np.any(graph<0):
        raise ValueError('Inverse-distance graphs require nonnegative distances')
    if graph_config.get('symmetric',False):
        graph = np.maximum(graph, graph.T)
    if graph_config.get('self_loops',False):
        np.fill_diagonal(graph, 1.)
    if not np.isfinite(graph).all():
        raise ValueError('Graph preprocessing produced non-finite adjacency weights')
    return dict(history=history, target=target, known=known, clock=clock,
                intervals=np.asarray(intervals), graph=graph, known_indices=np.asarray(known_indices, dtype=np.int64))


def prepare_traffic(data, training, evaluation, seed, logger=None):
    from libcity.data.traffic_cache import prepare_arrays
    folder = Path(data['data_dir']) / data['dataset']
    raw = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    raw = {**raw, **raw.get('info', {})}
    metadata = {key: value for key, value in raw.items() if key in METADATA_FIELDS}
    arrays, cache_info = prepare_arrays(data, metadata, lambda: _read_atomic(data, metadata, logger), logger,
        namespace='point-time-entity-v2')
    node_ids=pd.read_csv(folder/(metadata.get('geo_file',data['dataset'])+'.geo'))['geo_id'].tolist()
    history, target, known, clock, intervals, graph, known_indices = (arrays[name] for name in
        ('history', 'target', 'known', 'clock', 'intervals', 'graph', 'known_indices'))
    known_indices = known_indices.tolist()
    output_dim = target.shape[-1]
    cuts = boundaries(len(clock), data['split'], clock)
    scaler = fit_scaler(data['scaler']['name'], target[:cuts[0]])
    horizon, sampling = data['output_window'], data['sampling']
    offsets = np.arange(-data['input_window'], 0)
    layout = dict(kind='window', length=len(offsets))
    if sampling['name']=='crann':
        length = max(data['input_window'], sampling['temporal_history'])
        offsets = np.arange(-length, 0)
        layout = dict(kind='crann', length=length, temporal_history=sampling['temporal_history'])
    if sampling['name']=='cpt':
        from libcity.data.cpt_offsets import cpt_offsets
        offsets = cpt_offsets(sampling,metadata.get('time_intervals',300),horizon)
        maximum = int(-offsets.min())
        layout = dict(kind='cpt', length=len(offsets), max_position=maximum,
                      temporal_indices=(offsets+maximum).tolist(), sampling=dict(sampling))
    if sampling['name'] in ('frame_cpt','seasonal_windows'):
        from libcity.data.periodic_offsets import periodic_offsets
        offsets,layout = periodic_offsets(sampling,metadata.get('time_intervals',300),data['input_window'])
    layout.update(future_feature_indices=known_indices,
                  calendar=dict(time_in_day=known_indices[0] if data['features']['time_in_day'] else None,
                    day_of_week=known_indices[int(data['features']['time_in_day']):int(data['features']['time_in_day'])+7]
                    if data['features']['day_in_week'] else []),
                  observed_feature_indices=[i for i in range(history.shape[-1]) if i not in known_indices],
                  target_semantics='raw_and_scaled', split_semantics='target_time_v1',point_alignment='timestamp_entity_id_v2')
    starts = np.concatenate([np.arange(left-offsets.min(), right-horizon+1) for left,right in intervals])
    # A gap in a file is a boundary too: no window may span unobserved timestamps.
    interval = np.timedelta64(metadata.get('time_intervals',300), 's')
    gaps = np.r_[False,np.diff(clock)!=interval].cumsum()
    if sampling['name']=='seasonal_windows':
        # Service-frame shifts may cross overnight closures between windows.
        # Each actual segment and the recent-history/target span stay complete.
        complete = gaps[starts-data['input_window']]==gaps[starts+horizon-1]
        for segment in offsets.reshape(3,data['input_window']):
            complete &= gaps[starts+segment[0]]==gaps[starts+segment[-1]]
        starts = starts[complete]
        if gaps[-1]:
            # Affected old checkpoints must not silently resume with new samples.
            layout['gap_policy'] = 'seasonal_segments_v1'
    else:
        starts = starts[gaps[starts+horizon-1]==gaps[starts+offsets.min()]]
    selections = partition_windows(starts, horizon, cuts, len(clock))
    meta = DatasetMetadata(data['dataset'], history.shape[1], history.shape[-1], output_dim,
                           graph, {**metadata,'node_ids':node_ids}, layout)
    generator = torch.Generator().manual_seed(seed)
    loaders, split_info = [], {}
    for index, (name, selected) in enumerate(zip(('train','val','test'), selections)):
        if not len(selected):
            raise ValueError(f'{name} has no complete target windows; increase timeline or adjust windows/split')
        chosen = starts[selected]
        split_info[name] = dict(samples=len(chosen), target_start=str(clock[chosen.min()]),
            target_end=str(clock[chosen.max()+horizon-1]), target_index_start=int(chosen.min()),
            target_index_end=int(chosen.max()+horizon-1))
        if index==0 and data['pad_training']:
            padding = (-len(chosen)) % training['batch_size']
            chosen = np.pad(chosen, (0,padding), mode='edge')
        dataset = TrafficWindows(history,target,known,chosen,offsets,horizon,output_dim,scaler,clock)
        cfg = training if index==0 else evaluation
        loaders.append(DataLoader(dataset,batch_size=cfg['batch_size'],num_workers=cfg['num_workers'],
            shuffle=training['shuffle'] if index==0 else False, generator=generator if index==0 else None))
    summary = dict(train=len(loaders[0].dataset),val=len(loaders[1].dataset),test=len(loaders[2].dataset),
        train_padded=data['pad_training'],semantics='target_time_v1', input_window=len(offsets),
        requested_input_window=data['input_window'],adapter=layout['kind'], intervals=split_info,
        train_end=str(clock[cuts[0]]),val_end=str(clock[cuts[1]]),scaler_fit_end_exclusive=str(clock[cuts[0]]),
        discarded_boundary_windows=len(starts)-sum(map(len,selections)),
        history_span_points=int(-offsets.min()),cache=cache_info,
        sampling_gap_policy='seasonal_segments_and_recent_target_contiguous' if sampling['name']=='seasonal_windows' else 'full_input_span_contiguous')
    if logger:
        logger.info('Target-time split: %s', summary)
    return DataBundle(*loaders,meta,scaler,generator,summary)
