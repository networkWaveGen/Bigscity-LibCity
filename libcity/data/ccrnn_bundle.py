"""Train-interval SVD Gaussian graph initialization for coupled graph recurrence."""
from dataclasses import replace

import numpy as np
from scipy.spatial.distance import cdist

from libcity.data.timeline import boundaries
from libcity.data.traffic_bundle import prepare_traffic


def initial_graph(values,rank,normalization):
    values = np.asarray(values,dtype=np.float64)
    if values.ndim!=3 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError('CCRNN graph observations must be finite [training_steps,nodes,features]')
    if type(rank) is not int or rank<1:
        raise ValueError('CCRNN svd_rank must be a positive integer')
    if normalization not in ('randomwalk','laplacian'):
        raise ValueError('CCRNN normalization must be randomwalk or laplacian')
    matrix = values.transpose(0,2,1).reshape(-1,values.shape[1])
    _,singular,right = np.linalg.svd(matrix,full_matrices=False)
    actual = min(rank,len(singular))
    embedding = (singular[:actual,None]*right[:actual]).T
    distance = cdist(embedding,embedding)
    scale = float(distance.std())
    tolerance = np.finfo(np.float64).eps*max(float(np.linalg.norm(embedding)),1.)*64
    constant = bool(scale<=tolerance)
    similarity = np.ones_like(distance) if constant else np.exp(-distance/(scale*scale))
    np.fill_diagonal(similarity,0.)
    degree = similarity.sum(1)
    if normalization=='randomwalk':
        support = np.divide(similarity,degree[:,None],out=np.zeros_like(similarity),where=degree[:,None]>0)
    else:
        inverse = np.divide(1.,np.sqrt(degree),out=np.zeros_like(degree),where=degree>0)
        support = np.eye(len(degree))-inverse[:,None]*similarity*inverse[None,:]
    return support.astype(np.float32),dict(requested_rank=rank,actual_rank=actual,constant_distance=constant)


def prepare_ccrnn(data,training,evaluation,seed,logger=None):
    options = data['graph']
    # SVD settings are split-dependent transforms, not raw-array cache identity.
    plain = {**data,'graph':dict(name='none',symmetric=False,self_loops=False)}
    bundle = prepare_traffic(plain,training,evaluation,seed,logger)
    meta,dataset = bundle.metadata,bundle.train.dataset
    channels = options['svd_channels']
    if channels is None:
        columns = meta.info.get('data_col')
        width = 1 if isinstance(columns,str) else len(columns) if isinstance(columns,(tuple,list)) else None
        channels = list(range(width)) if width is not None else list(meta.input_layout['observed_feature_indices'])
    if not set(channels)<=set(meta.input_layout['observed_feature_indices']):
        raise ValueError('CCRNN svd_channels must select historical observations, not known calendar/future features')
    end = boundaries(len(dataset.clock),data['split'],dataset.clock)[0]
    values = dataset.history[:end,:,channels]
    support,details = initial_graph(values,options['svd_rank'],options['normalization'])
    description = dict(source='raw_training_interval',start_index=0,end_index_exclusive=end,
        first_timestamp=str(dataset.clock[0]),last_timestamp=str(dataset.clock[end-1]),channels=channels,
        observations_shape=list(values.shape),normalization=options['normalization'],
        similarity='exp(-euclidean/std(distance)^2)',**details)
    bundle.metadata = replace(meta,adjacency=support,input_layout={**meta.input_layout,'graph_initialization':description})
    bundle.split_summary['graph_initialization'] = description
    return bundle
