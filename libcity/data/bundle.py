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

    def __post_init__(self):
        object.__setattr__(self, 'adjacency', readonly(self.adjacency))
        object.__setattr__(self, 'info', readonly(dict(self.info)))


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


def prepare_traffic(data_config, training_config, evaluation_config, seed):
    from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
    folder = Path(data_config['data_dir']) / data_config['dataset']
    raw = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    raw = {**raw, **raw.get('info', {})}
    metadata = {key: value for key, value in raw.items() if key in METADATA_FIELDS}
    legacy = dict(metadata, dataset=data_config['dataset'], data_dir=data_config['data_dir'],
        cache_dir=data_config['cache_dir'], cache_dataset=data_config['cache'],
        input_window=data_config['input_window'], output_window=data_config['output_window'],
        train_rate=data_config['split']['train'], eval_rate=data_config['split']['val'],
        scaler=data_config['scaler']['name'], batch_size=training_config['batch_size'], num_workers=0,
        pad_with_last_sample=data_config['pad_training'], load_external=data_config['features']['load_external'],
        add_time_in_day=data_config['features']['time_in_day'], add_day_in_week=data_config['features']['day_in_week'])
    dataset = TrafficStatePointDataset(legacy)
    loaders = dataset.get_data()
    features = dataset.get_data_feature()
    meta = DatasetMetadata(data_config['dataset'], features['num_nodes'], features['feature_dim'],
                           features['output_dim'], features['adj_mx'], metadata)
    generator = torch.Generator().manual_seed(seed)
    converted = []
    for index, old in enumerate(loaders):
        options = training_config if index == 0 else evaluation_config
        converted.append(DataLoader(DictDataset(old.dataset, dataset.feature_name),
            batch_size=options['batch_size'], num_workers=options['num_workers'],
            shuffle=training_config['shuffle'] if index == 0 else False,
            generator=generator if index == 0 else None))
    summary = dict(train=len(converted[0].dataset), val=len(converted[1].dataset), test=len(converted[2].dataset),
                   train_padded=data_config['pad_training'], semantics='legacy_window_count')
    return DataBundle(*converted, meta, dataset.scaler, generator, summary)
