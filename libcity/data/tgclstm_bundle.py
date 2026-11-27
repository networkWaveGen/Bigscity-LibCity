"""A lightweight FFR edge adapter over the common atomic time-series bundle."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from libcity.data.traffic_bundle import prepare_traffic
from libcity.data.traffic_cache import source_signature


def prepare_tgclstm(data,training,evaluation,seed,logger=None):
    folder = Path(data['data_dir'])/data['dataset']
    sources = [folder/'config.json',*sorted(folder.glob('*.geo')),*sorted(folder.glob('*.rel'))]
    before = source_signature(sources)
    bundle = prepare_traffic(data,training,evaluation,seed,logger)
    column = data['graph'].get('reachability_column')
    if column is None:
        return bundle
    metadata = bundle.metadata
    geo = pd.read_csv(folder/(metadata.info.get('geo_file',data['dataset'])+'.geo'))
    edges = pd.read_csv(folder/(metadata.info.get('rel_file',data['dataset'])+'.rel'))
    if column not in edges.columns:
        raise ValueError(f'TGCLSTM reachability column {column!r} missing from .rel; select an existing column or explicit null')
    indices = {key:index for index,key in enumerate(geo['geo_id'])}
    if len(indices)!=metadata.num_nodes:
        raise ValueError('TGCLSTM geo IDs must be unique and match the traffic node count')
    reachability = np.zeros((metadata.num_nodes,metadata.num_nodes),dtype=np.float32)
    edges = edges.assign(**{column:pd.to_numeric(edges[column],errors='coerce')})
    for origin,destination,value in edges[['origin_id','destination_id',column]].itertuples(index=False,name=None):
        if origin not in indices or destination not in indices:
            continue
        if not np.isfinite(value) or value<0:
            raise ValueError('TGCLSTM reachability must be finite and nonnegative on listed edges')
        reachability[indices[origin],indices[destination]] = value
    if source_signature(sources)!=before:
        raise ValueError('TGCLSTM graph sources changed during preprocessing; retry with stable inputs')
    bundle.metadata = replace(metadata,input_layout={**metadata.input_layout,'reachability':reachability,
        'reachability_column':column})
    bundle.split_summary['graph_reachability'] = dict(column=column,nonzero_edges=int(np.count_nonzero(reachability)),
        missing_edge_value=0.)
    return bundle
