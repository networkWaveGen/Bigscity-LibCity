"""Observed relation distances, without confusing missing edges and zero distance."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from libcity.data.traffic_bundle import prepare_traffic
from libcity.data.traffic_cache import source_signature


def partition_distances(distances):
    distances = np.array(distances,dtype=np.float64,copy=True)
    if distances.ndim!=2 or distances.shape[0]!=distances.shape[1] or np.isnan(distances).any() or np.any(distances<0):
        raise ValueError('Distance bands require square nonnegative distances; positive infinity means no edge')
    bands = np.zeros((3,*distances.shape),dtype=np.float32)
    for row,values in enumerate(distances):
        selected = np.flatnonzero(np.isfinite(values))
        if len(selected):
            finite = values[selected]
            span = finite.max()-finite.min()
            groups = np.minimum(2,np.floor((finite-finite.min())/span*3)).astype(int) if span>0 else np.zeros(len(selected),dtype=int)
            bands[groups,row,selected] = 1
    return bands


def prepare_distance_bands(data,training,evaluation,seed,logger=None):
    folder = Path(data['data_dir'])/data['dataset']
    raw = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    raw = {**raw,**raw.get('info',{})}
    geo_path = folder/(raw.get('geo_file',data['dataset'])+'.geo')
    rel_path = folder/(raw.get('rel_file',data['dataset'])+'.rel')
    before = source_signature([folder/'config.json',geo_path,rel_path])
    geo,rel = pd.read_csv(geo_path),pd.read_csv(rel_path)
    ids = geo['geo_id'].tolist()
    if geo['geo_id'].isna().any() or len(set(ids))!=len(ids):
        raise ValueError('Distance-band geo IDs must be distinct and nonmissing')
    mapping = {value:index for index,value in enumerate(ids)}
    binary = data['graph']['name']=='binary'
    column = data['graph'].get('distance_column') or raw.get('weight_col')
    if isinstance(column,list):
        column = column[0] if len(column)==1 else None
    if column is None:
        candidates = [name for name in rel if name not in ('rel_id','type','origin_id','destination_id')]
        column = candidates[0] if len(candidates)==1 else None
    if not binary and (column is None or column not in rel):
        raise ValueError('MultiSTGCnet distance_bands requires distance_column or one unambiguous .rel distance property')
    distances = np.full((len(ids),len(ids)),np.inf,dtype=np.float64)
    for row in rel.to_dict('records'):
        origin,destination = row['origin_id'],row['destination_id']
        if origin not in mapping or destination not in mapping:
            raise ValueError('Distance-band relation endpoint is absent from .geo')
        value = 1. if binary else float(row[column])
        if not np.isfinite(value) or value<0:
            raise ValueError('Listed relation distances must be finite and nonnegative')
        distances[mapping[origin],mapping[destination]] = value
    if data['graph'].get('symmetric',False):
        distances = np.minimum(distances,distances.T)
    if data['graph'].get('self_loops',False):
        np.fill_diagonal(distances,0.)
    bundle = prepare_traffic(data,training,evaluation,seed,logger)
    if len(ids)!=bundle.metadata.num_nodes:
        raise ValueError('Distance-band nodes do not match traffic observations')
    if source_signature([folder/'config.json',geo_path,rel_path])!=before:
        raise ValueError('Distance-band sources changed during preprocessing; retry with stable inputs')
    matrices = partition_distances(distances)
    bundle.metadata = replace(bundle.metadata,input_layout={**bundle.metadata.input_layout,
        'distance_bands':matrices,'distance_node_ids':ids,'distance_column':None if binary else column})
    bundle.split_summary['distance_bands'] = dict(column=None if binary else column,
        semantics='listed_edges_rowwise_equal_distance_ranges',constant_row_band='near',
        counts={name:int(matrix.sum()) for name,matrix in zip(('near','middle','distant'),matrices)},
        symmetric_distance='minimum_observed_direction' if data['graph'].get('symmetric',False) else False,
        include_upper_bound=True,missing_edges='excluded')
    if logger:
        logger.info('Distance bands: %s',bundle.split_summary['distance_bands'])
    return bundle
