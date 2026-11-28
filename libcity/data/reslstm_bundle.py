"""ResLSTM distinguishes raw flow observations from historical global covariates."""
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from libcity.data.traffic_bundle import prepare_traffic
from libcity.data.traffic_cache import source_signature


def prepare_reslstm(data,training,evaluation,seed,logger=None):
    folder = Path(data['data_dir'])/data['dataset']
    path = folder/'config.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    raw = {**raw,**raw.get('info',{})}
    files = raw.get('data_files',data['dataset'])
    files = [files] if isinstance(files,str) else files
    sources = [path,*[folder/(name+'.dyna') for name in files]]
    before = source_signature(sources)
    columns = raw.get('data_col')
    columns = [columns] if isinstance(columns,str) and columns else columns
    if not columns:
        reserved = {'dyna_id','type','time','entity_id'}
        columns = [name for name in pd.read_csv(sources[1],nrows=0) if name not in reserved]
    if (not isinstance(columns,list) or len(columns)<2 or any(not isinstance(name,str) or not name for name in columns)
            or len(set(columns))!=len(columns)):
        raise ValueError('ResLSTM requires at least two distinct raw flow observation columns')
    bundle = prepare_traffic(data,training,evaluation,seed,logger)
    if len(columns)>bundle.metadata.feature_dim:
        raise ValueError('ResLSTM raw observation columns exceed the prepared input width')
    if source_signature(sources)!=before:
        raise ValueError('ResLSTM source schema changed during preprocessing; retry with stable inputs')
    indices = list(range(len(columns),bundle.metadata.feature_dim))
    bundle.metadata = replace(bundle.metadata,input_layout={**bundle.metadata.input_layout,
        'reslstm_observation_dim':len(columns),'reslstm_external_indices':indices})
    bundle.split_summary['reslstm_inputs'] = dict(observation_columns=columns,
        historical_global_external_indices=indices,future_external_inputs=False)
    return bundle
