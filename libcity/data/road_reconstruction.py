"""Transductive road attributes: shared graph inputs and disjoint supervised nodes."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from libcity.data.bundle import DataBundle, DatasetMetadata
from libcity.data.task_bundles import atomic
from libcity.data.timeline import boundaries
from libcity.utils.normalization import NoneScaler


class FeatureScaler:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.std = np.where(self.std > 1e-8, self.std, 1.)

    def parameters(self, values):
        if isinstance(values, torch.Tensor):
            return (torch.as_tensor(value, dtype=values.dtype, device=values.device) for value in (self.mean,self.std))
        return self.mean,self.std

    def transform(self, values):
        mean,std = self.parameters(values)
        return (values-mean)/std

    def inverse_transform(self, values):
        mean,std = self.parameters(values)
        return values*std+mean


class GraphNodes(Dataset):
    def __init__(self, features, targets, raw, indices):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets[indices], dtype=torch.float32)
        self.raw = torch.tensor(raw[indices], dtype=torch.float32)
        self.indices = torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return dict(node_features=self.features.clone(), node_indices=self.indices.clone(),
                    y=self.targets.clone(), y_raw=self.raw.clone())


def single_graph(rows):
    return rows[0]


def prepare_reconstruction(data, training, evaluation, seed, logger=None):
    folder,raw = atomic(data)
    geo = pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    rel = pd.read_csv(folder/(raw.get('rel_file',data['dataset'])+'.rel'))
    ids = geo['geo_id'].tolist()
    if len(ids)<3 or len(set(ids))!=len(ids):
        raise ValueError('Road reconstruction needs at least three distinct node IDs')
    mapping = {value:index for index,value in enumerate(ids)}
    edges = set()
    for origin,destination in zip(rel['origin_id'],rel['destination_id']):
        if origin not in mapping or destination not in mapping:
            raise ValueError('Road relation endpoint is absent from .geo')
        edges.add((mapping[origin],mapping[destination]))
    edge_index = np.asarray(sorted(edges),dtype=np.int64).reshape(-1,2).T
    columns = data['feature_columns'] or [name for name in geo if name not in ('geo_id','type','coordinates')]
    if not columns or any(name not in geo for name in columns):
        raise ValueError('Road reconstruction requires existing .geo attribute feature_columns')
    arrays,names,categories = [],[],{}
    for name in columns:
        series = geo[name]
        if name in data['categorical_columns'] or not pd.api.types.is_numeric_dtype(series):
            values = series.fillna('<missing>').astype(str)
            vocabulary = sorted(values.unique().tolist())
            arrays.extend((values==value).to_numpy(dtype=np.float64)[:,None] for value in vocabulary)
            names.extend(f'{name}={value}' for value in vocabulary)
            categories[name] = vocabulary
        else:
            values = series.to_numpy(dtype=np.float64)
            if np.isinf(values).any():
                raise ValueError('Road numeric attributes must not contain infinities')
            arrays.append(values[:,None])
            names.append(name)
    features = np.concatenate(arrays,axis=1)
    order = np.random.default_rng(seed).permutation(len(ids))
    partitions = [np.sort(part) for part in np.split(order,boundaries(len(ids),data['split']))]
    if any(not len(part) for part in partitions):
        raise ValueError('Road node splits must all be nonempty')
    observed = features[partitions[0]]
    if np.any(np.isfinite(observed).sum(axis=0)==0):
        raise ValueError('Each road feature needs a finite training-node observation')
    mean,std = np.nanmean(observed,axis=0),np.nanstd(observed,axis=0)
    inputs = np.where(np.isfinite(features),features,mean)
    scaler = FeatureScaler(mean,std) if data['scaler']['name']=='standard' else NoneScaler()
    targets = scaler.transform(features)
    inputs = scaler.transform(inputs)
    metadata = DatasetMetadata(data['dataset'],len(ids),features.shape[1],features.shape[1],
        info=dict(node_ids=ids,feature_names=names,categories=categories,edge_index=edge_index),
        input_layout=dict(kind='node_reconstruction',graph='transductive',scaler_fit='training_nodes'))
    generator = torch.Generator().manual_seed(seed)
    loaders = []
    for index,indices in enumerate(partitions):
        cfg = training if index==0 else evaluation
        loaders.append(DataLoader(GraphNodes(inputs,targets,features,indices),batch_size=1,
            num_workers=cfg['num_workers'],collate_fn=single_graph,generator=generator if index==0 else None))
    summary = dict(semantics='transductive_node_reconstruction',full_graph_batch=True,
        train=len(partitions[0]),val=len(partitions[1]),test=len(partitions[2]),
        node_indices={name:indices.tolist() for name,indices in zip(('train','val','test'),partitions)},
        feature_names=names,scaler_fit='training_nodes')
    if logger:
        logger.info('Road reconstruction protocol: %s',summary)
    return DataBundle(*loaders,metadata,scaler,generator,summary)
