"""Sparse full-graph inputs and bounded, replayable random-walk sampling."""
from collections import OrderedDict
import random

import numpy as np
import pandas as pd

from libcity.data.bundle import DataBundle,DatasetMetadata
from libcity.data.task_bundles import atomic


def prepare_walk_graph(data,training,evaluation,seed,logger=None):
    folder,raw = atomic(data)
    geo = pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    if 'geo_id' not in geo or geo.empty or geo.geo_id.isna().any() or geo.geo_id.duplicated().any():
        raise ValueError('Walk embeddings require nonempty distinct .geo IDs')
    ids = geo.geo_id.tolist()
    mapping = {identity:index for index,identity in enumerate(ids)}
    relations = pd.read_csv(folder/(raw.get('rel_file',data['dataset'])+'.rel'))
    if not {'origin_id','destination_id'}<=set(relations):
        raise ValueError('Walk embeddings require .rel origin_id/destination_id')
    graph = data['graph']
    column = graph['weight_column'] or raw.get('weight_col','weight')
    if graph['name']=='weighted' and column not in relations:
        raise ValueError('Weighted walk graph requires its declared weight column')
    edges = {}
    for row in relations.to_dict('records'):
        if row['origin_id'] not in mapping or row['destination_id'] not in mapping:
            raise ValueError('Walk relation endpoint is absent from .geo')
        origin,destination = mapping[row['origin_id']],mapping[row['destination_id']]
        weight = float(row[column]) if graph['name']=='weighted' else 1.
        if not np.isfinite(weight) or weight<0:
            raise ValueError('Walk edge weights must be finite and nonnegative')
        if origin==destination and not graph['self_loops']:
            continue
        # A repeated directed relation replaces the previous matrix entry.
        edges[origin,destination] = weight
    edges = {edge:weight for edge,weight in edges.items() if weight>0}
    if not graph['directed']:
        for (origin,destination),weight in list(edges.items()):
            weight = max(weight,edges.get((destination,origin),0.))
            edges[origin,destination] = edges[destination,origin] = weight
    index = np.asarray(sorted(edges),dtype=np.int64).reshape(-1,2).T
    weights = np.asarray([edges[tuple(edge)] for edge in index.T],dtype=np.float64)
    offsets = np.r_[0,np.cumsum(np.bincount(index[0],minlength=len(ids)))]
    isolated = int(np.count_nonzero(np.bincount(index.reshape(-1),minlength=len(ids))==0))
    dead_ends = int(np.count_nonzero(np.diff(offsets)==0))
    info = dict(node_ids=ids,edge_index=index,edge_weights=weights,row_offsets=offsets,
        graph_kind=graph['name'],directed=graph['directed'],isolated_nodes=isolated,dead_end_nodes=dead_ends)
    metadata = DatasetMetadata(data['dataset'],num_nodes=len(ids),info=info,
        input_layout=dict(kind='walk_embedding',protocol='transductive_full_graph',directed=graph['directed'],
            graph_kind=graph['name'],self_loops=graph['self_loops']))
    summary = dict(semantics='transductive_full_graph_no_holdout',nodes=len(ids),edges=len(weights),
        isolated_nodes=isolated,dead_end_nodes=dead_ends,directed=graph['directed'],graph_kind=graph['name'])
    return DataBundle(None,None,None,metadata,split_summary=summary)


def alias_table(probabilities):
    count = len(probabilities)
    threshold = np.asarray(probabilities,dtype=np.float64)*count
    aliases = np.zeros(count,dtype=np.int64)
    small = [index for index,value in enumerate(threshold) if value<1.]
    large = [index for index,value in enumerate(threshold) if value>=1.]
    while small and large:
        left,right = small.pop(),large.pop()
        aliases[left] = right
        threshold[right] += threshold[left]-1.
        (small if threshold[right]<1. else large).append(right)
    threshold[small+large] = 1.
    return aliases,threshold


class WalkCorpus:
    def __init__(self,metadata,*,algorithm,num_walks,walk_length,seed,p=1.,q=1.,restart=0.,cache_budget=100000):
        info = metadata.info
        self.edges,self.weights,self.offsets = info['edge_index'],info['edge_weights'],info['row_offsets']
        self.algorithm,self.count,self.length,self.seed = algorithm,num_walks,walk_length,seed
        self.p,self.q,self.restart = p,q,restart
        self.cache,self.cache_size,self.cache_budget = OrderedDict(),0,cache_budget

    def neighbours(self,node):
        return self.edges[1,self.offsets[node]:self.offsets[node+1]]

    def has_edge(self,origin,destination):
        neighbours = self.neighbours(origin)
        position = np.searchsorted(neighbours,destination)
        return position<len(neighbours) and neighbours[position]==destination

    def probabilities(self,previous,current):
        neighbours = self.neighbours(current)
        weights = self.weights[self.offsets[current]:self.offsets[current+1]]
        if not len(neighbours): return weights
        logits = np.log(weights)
        if previous>=0:
            divisors = np.asarray([self.p if node==previous else 1. if self.has_edge(int(node),previous) else self.q for node in neighbours])
            logits = logits-np.log(divisors)
        values = np.exp(logits-logits.max())
        return values/values.sum()

    def table(self,previous,current):
        key = (previous,current)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        result = alias_table(self.probabilities(previous,current))
        size = len(result[0])
        if 0<size<=self.cache_budget:
            while self.cache and self.cache_size+size>self.cache_budget:
                _,old = self.cache.popitem(last=False);self.cache_size -= len(old[0])
            self.cache[key] = result;self.cache_size += size
        return result

    def __iter__(self):
        python = random.Random(self.seed)
        numpy = np.random.RandomState(self.seed)
        nodes = list(range(len(self.offsets)-1))
        for _ in range(self.count):
            python.shuffle(nodes)
            for start in nodes:
                path = [start]
                while len(path)<self.length:
                    current = path[-1]
                    neighbours = self.neighbours(current)
                    if not len(neighbours): break
                    if self.algorithm=='deepwalk':
                        node = int(python.choice(neighbours)) if python.random()>=self.restart else start
                    else:
                        aliases,threshold = self.table(path[-2] if len(path)>1 else -1,current)
                        index = int(numpy.rand()*len(aliases))
                        node = int(neighbours[index if numpy.rand()<threshold[index] else aliases[index]])
                    path.append(node)
                yield [str(node) for node in path]
