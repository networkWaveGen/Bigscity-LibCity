"""Overlapping biconnected regions keep real node identities and probability targets."""
from dataclasses import replace
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from libcity.data.traffic_bundle import prepare_traffic


OPTIONS={'region_threshold','region_groups','reconstruction'}


def region_mapping(graph,ids,threshold=.8,groups=None):
    graph=np.asarray(graph,dtype=np.float64)
    if graph.shape!=(len(ids),len(ids)) or not len(ids) or not np.isfinite(graph).all() or np.any(graph<0):
        raise ValueError('HIEST regions require a finite nonnegative graph aligned to node IDs')
    if not np.isfinite(threshold) or threshold<0: raise ValueError('HIEST region_threshold must be nonnegative and finite')
    index={value:position for position,value in enumerate(ids)}
    if len(index)!=len(ids): raise ValueError('HIEST node IDs must be distinct')
    if groups is None:
        edges=np.argwhere(np.triu((graph>threshold)|(graph.T>threshold),k=1))
        network=nx.Graph();network.add_nodes_from(range(len(ids)));network.add_edges_from(map(tuple,edges))
        components=[set(group) for group in nx.biconnected_components(network)]
        covered=set().union(*components) if components else set()
        components.extend({node} for node in range(len(ids)) if node not in covered)
        # Physical IDs make the automatic region order independent of .geo row order.
        components.sort(key=lambda group:tuple(sorted(repr(ids[node]) for node in group)))
    else:
        components=[]
        if not isinstance(groups,list) or not groups: raise ValueError('HIEST region_groups must be nonempty groups')
        try:
            for group in groups:
                if not isinstance(group,list) or not group or len(set(group))!=len(group):
                    raise ValueError('HIEST each region requires distinct real node IDs')
                components.append({index[value] for value in group})
        except (KeyError,TypeError) as error:
            raise ValueError('HIEST region contains an unknown node ID') from error
        if len({frozenset(group) for group in components})!=len(components) or set().union(*components)!=set(range(len(ids))):
            raise ValueError('HIEST explicit regions must be distinct and cover every node')
    mapping=np.zeros((len(ids),len(components)),dtype=np.float64)
    for column,group in enumerate(components): mapping[list(group),column]=1./len(group)
    return mapping,[[ids[node] for node in sorted(group,key=lambda node:repr(ids[node]))] for group in components]


def prepare_hiest(data,training,evaluation,seed,logger=None):
    if data['sampling']['name']!='window': raise ValueError('HIEST requires continuous windows')
    options=data['graph']
    plain=dict(data,graph={key:value for key,value in options.items() if key not in OPTIONS})
    bundle=prepare_traffic(plain,training,evaluation,seed,logger)
    folder=Path(data['data_dir'])/data['dataset']
    raw=json.loads((folder/'config.json').read_text(encoding='utf-8'));raw={**raw,**raw.get('info',{})}
    geo=pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    ids=geo['geo_id'].tolist()
    if geo['geo_id'].isna().any() or (pd.api.types.is_numeric_dtype(geo['geo_id']) and not np.isfinite(geo['geo_id']).all()):
        raise ValueError('HIEST node IDs must be finite')
    mapping,groups=region_mapping(bundle.metadata.adjacency,ids,options['region_threshold'],options['region_groups'])
    graph=np.array(bundle.metadata.adjacency,dtype=np.float64,copy=True)
    target=(graph>0).astype(np.float64) if options['reconstruction']=='binary' else graph.copy()
    if options['reconstruction'] not in ('binary','weights') or np.any(target>1):
        raise ValueError('HIEST reconstruction weights must be probabilities in [0,1], or select binary targets')
    region_target=np.clip(mapping.T@target@mapping,0.,1.)
    description=dict(method='biconnected_components' if options['region_groups'] is None else 'explicit_overlapping_groups',
        threshold=options['region_threshold'],region_node_ids=groups,regional_nodes=mapping.shape[1],
        mapping='column_normalized_membership',reconstruction=options['reconstruction'])
    bundle.metadata=replace(bundle.metadata,info={**bundle.metadata.info,'node_ids':ids},
        input_layout={**bundle.metadata.input_layout,'hierarchy':description,'input_missing_observations':'raw_zero_before_scaling'},
        static_data={**bundle.metadata.static_data,'region_membership':mapping.astype(np.float32),
            'original_reconstruction':target.astype(np.float32),'regional_reconstruction':region_target.astype(np.float32)})
    # Point windows keep raw targets separately, so only historical inputs are imputed.
    history=bundle.train.dataset.history
    if np.isinf(history).any(): raise ValueError('HIEST historical observations must not contain infinities')
    cleaned=np.nan_to_num(history,nan=0.)
    for loader in (bundle.train,bundle.val,bundle.test): loader.dataset.history=cleaned
    bundle.split_summary.update(hierarchy=description,input_missing_observations='raw_zero_before_scaling')
    if logger: logger.info('HIEST static overlapping hierarchy: %s',description)
    return bundle
