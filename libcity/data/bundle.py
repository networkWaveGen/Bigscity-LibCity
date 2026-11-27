"""Public data contracts and a small bridge to existing atomic traffic loaders."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def readonly(value):
    if isinstance(value, dict):
        return MappingProxyType({key: readonly(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(readonly(item) for item in value)
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    return value


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    num_nodes: int = 0
    feature_dim: int = 0
    output_dim: int = 0
    adjacency: np.ndarray | None = field(default=None, repr=False, compare=False)
    info: object = field(default_factory=dict)
    input_layout: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'adjacency', readonly(self.adjacency))
        object.__setattr__(self, 'info', readonly(dict(self.info)))
        object.__setattr__(self, 'input_layout', readonly(dict(self.input_layout)))


@dataclass
class DataBundle:
    train: object
    val: object
    test: object
    metadata: DatasetMetadata
    scaler: object = None
    generator: torch.Generator | None = None
    split_summary: dict = field(default_factory=dict)


class DictDataset(Dataset):
    def __init__(self, samples, features):
        self.samples, self.features = samples, features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {key: torch.tensor(value, dtype=torch.float32 if kind == 'float' else torch.long)
                for (key, kind), value in zip(self.features.items(), self.samples[index], strict=True)}


METADATA_FIELDS = {'geo_file', 'rel_file', 'data_files', 'ext_file', 'data_col', 'ext_col', 'weight_col',
                   'output_dim', 'time_intervals'}


def prepare_traffic(data_config, training_config, evaluation_config, seed, logger=None):
    from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
    folder = Path(data_config['data_dir']) / data_config['dataset']
    raw = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    raw = {**raw, **raw.get('info', {})}
    metadata = {key: value for key, value in raw.items() if key in METADATA_FIELDS}
    features_config, graph_config = data_config['features'], data_config['graph']
    legacy = dict(metadata, dataset=data_config['dataset'], data_dir=data_config['data_dir'],
        cache_dir=data_config['cache_dir'], cache_dataset=data_config['cache'],
        input_window=data_config['input_window'], output_window=data_config['output_window'],
        train_rate=data_config['split']['train'], eval_rate=data_config['split']['val'],
        scaler=data_config['scaler']['name'], batch_size=training_config['batch_size'], num_workers=0,
        pad_with_last_sample=data_config['pad_training'],
        load_external=any(features_config.values()),
        include_external=features_config['load_external'],
        add_time_in_day=features_config['time_in_day'], add_day_in_week=features_config['day_in_week'],
        init_weight_inf_or_zero='inf' if graph_config['name']=='distance' else 'zero',
        set_weight_link_or_dist='link' if graph_config['name']=='binary' else 'dist',
        calculate_weight_adj=graph_config['name']=='distance',weight_adj_epsilon=graph_config.get('epsilon',.1))
    sampling = data_config['sampling']
    layout = dict(kind='window',length=data_config['input_window'])
    dataset_class = TrafficStatePointDataset
    if sampling['name']=='cpt':
        from libcity.data.dataset.dataset_subclass.astgcn_dataset import ASTGCNDataset
        dataset_class = ASTGCNDataset
        for key in ('closeness','period','trend'):
            legacy['len_'+key] = sampling.get(key,0)
        for key in ('interval_period','interval_trend'):
            legacy[key] = sampling.get(key,1)
        points = 3600//metadata.get('time_intervals',300)
        horizon = data_config['output_window']
        if points<1 or horizon>points:
            raise ValueError('CPT adapter requires output_window <= points per hour to keep history before targets')
        segments = [(sampling.get('trend',0),7*24*sampling.get('interval_trend',1)),
                    (sampling.get('period',0),24*sampling.get('interval_period',1)),
                    (sampling.get('closeness',0),1)]
        maximum = max(count*points*units for count,units in segments)
        indices = [maximum-depend*points*units+index for count,units in segments
                   for depend in range(1,count+1) for index in range(horizon)]
        layout = dict(kind='cpt',length=len(indices),max_position=maximum,temporal_indices=indices,sampling=dict(sampling))
    dataset = dataset_class(legacy, logger=logger)
    loaders = dataset.get_data()
    features = dataset.get_data_feature()
    graph = features['adj_mx'].copy()
    if graph_config.get('symmetric',False):
        graph = np.maximum(graph,graph.T)
    if graph_config.get('self_loops',False):
        np.fill_diagonal(graph,1.)
    if not np.isfinite(graph).all():
        raise ValueError('Graph preprocessing produced non-finite adjacency weights')
    meta = DatasetMetadata(data_config['dataset'], features['num_nodes'], features['feature_dim'],
                           features['output_dim'], graph, metadata, layout)
    generator = torch.Generator().manual_seed(seed)
    converted = []
    for index, old in enumerate(loaders):
        options = training_config if index == 0 else evaluation_config
        converted.append(DataLoader(DictDataset(old.dataset, dataset.feature_name),
            batch_size=options['batch_size'], num_workers=options['num_workers'],
            shuffle=training_config['shuffle'] if index == 0 else False,
            generator=generator if index == 0 else None))
    summary = dict(train=len(converted[0].dataset), val=len(converted[1].dataset), test=len(converted[2].dataset),
                   train_padded=data_config['pad_training'], semantics='legacy_window_count',
                   input_window=layout['length'],requested_input_window=data_config['input_window'],adapter=layout['kind'])
    return DataBundle(*converted, meta, dataset.scaler, generator, summary)
