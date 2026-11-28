"""Static, reproducible graph partitions over the common traffic windows."""
from dataclasses import replace

import numpy as np

from libcity.data.traffic_bundle import prepare_traffic

HIERARCHY_OPTIONS = {'cluster_nodes','cluster_labels','cluster_epsilon'}


def canonical_labels(labels):
    mapping = {}
    result = []
    for label in labels:
        mapping.setdefault(int(label),len(mapping))
        result.append(mapping[int(label)])
    return np.asarray(result,dtype=np.int64)


def graph_partition(graph, count, seed, labels=None):
    graph = np.asarray(graph)
    if graph.ndim!=2 or not len(graph) or graph.shape[0]!=graph.shape[1] or not np.isfinite(graph).all() or np.any(graph<0):
        raise ValueError('HGCN requires a finite nonnegative square graph for clustering')
    nodes = len(graph)
    if type(count) is not int or count<0:
        raise ValueError('HGCN cluster_nodes must be a nonnegative integer')
    if labels is not None:
        if len(labels)!=nodes or any(type(value) is not int or value<0 for value in labels):
            raise ValueError('HGCN cluster_labels must assign a nonnegative integer to each geo-ordered node')
        labels = canonical_labels(labels)
        if count and len(np.unique(labels))!=count:
            raise ValueError('HGCN cluster_nodes does not match explicit cluster_labels')
        return labels
    count = min(10,nodes) if count==0 else count
    if type(count) is not int or not 1<=count<=nodes:
        raise ValueError('HGCN cluster_nodes must be zero (automatic) or between one and num_nodes')
    if count==1:
        return np.zeros(nodes,dtype=np.int64)
    if count==nodes:
        return np.arange(nodes,dtype=np.int64)
    affinity = (graph.astype(np.float64)+graph.T)/2
    np.fill_diagonal(affinity,0.)
    if not np.any(affinity):
        return np.arange(nodes,dtype=np.int64)%count
    from sklearn.cluster import SpectralClustering
    labels = SpectralClustering(n_clusters=count,affinity='precomputed',assign_labels='discretize',
        random_state=seed).fit_predict(affinity)
    labels = canonical_labels(labels)
    if len(np.unique(labels))!=count:
        raise ValueError('HGCN spectral partition has empty groups; provide explicit cluster_labels')
    return labels


def cluster_graph(graph, membership, epsilon):
    aggregate = membership.T@np.asarray(graph,dtype=np.float64)@membership
    deviation = aggregate.std()
    # Retain LibCity's Gaussian of summed graph weights, with a neutral zero
    # graph when its scale is undefined instead of introducing NaN edges.
    result = np.exp(-np.square(aggregate/deviation)) if deviation>0 else np.zeros_like(aggregate)
    result[result<epsilon] = 0.
    return result.astype(np.float32)


def prepare_hgcn(data,training,evaluation,seed,logger=None):
    options = data['graph']
    plain = {**data,'graph':{key:value for key,value in options.items() if key not in HIERARCHY_OPTIONS}}
    bundle = prepare_traffic(plain,training,evaluation,seed,logger)
    if bundle.metadata.input_layout['kind']!='window':
        raise ValueError('HGCN requires continuous window sampling')
    labels = graph_partition(bundle.metadata.adjacency,options.get('cluster_nodes',0),seed,options.get('cluster_labels'))
    membership = np.eye(int(labels.max())+1,dtype=np.float32)[labels]
    epsilon = options.get('cluster_epsilon',.1)
    region_graph = cluster_graph(bundle.metadata.adjacency,membership,epsilon)
    hierarchy = dict(membership=membership,cluster_adjacency=region_graph,labels=labels.tolist(),
        clustering='explicit' if options.get('cluster_labels') is not None else 'spectral_or_degenerate',
        seed=seed if options.get('cluster_labels') is None else None,cluster_epsilon=epsilon,
        cluster_graph='gaussian_of_summed_weights',node_aggregation='sum_all_members')
    bundle.metadata = replace(bundle.metadata,input_layout={**bundle.metadata.input_layout,'hierarchy':hierarchy})
    bundle.split_summary['hierarchy'] = {key:value for key,value in hierarchy.items() if key not in ('membership','cluster_adjacency')}
    bundle.split_summary['hierarchy']['cluster_nodes'] = membership.shape[1]
    return bundle
