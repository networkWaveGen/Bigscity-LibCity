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
    static_data: object = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'adjacency', readonly(self.adjacency))
        object.__setattr__(self, 'info', readonly(dict(self.info)))
        object.__setattr__(self, 'input_layout', readonly(dict(self.input_layout)))
        arrays = dict(self.static_data)
        if any(not isinstance(key,str) or not isinstance(value,np.ndarray) for key,value in arrays.items()):
            raise TypeError('DatasetMetadata.static_data requires named NumPy arrays')
        object.__setattr__(self, 'static_data', readonly(arrays))


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
    from libcity.data.traffic_bundle import prepare_traffic as prepare
    return prepare(data_config, training_config, evaluation_config, seed, logger)
