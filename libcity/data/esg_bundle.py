"""ESG graph initialization uses a declared, original training observation prefix."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset

from libcity.data.grid_traffic import prepare_grid
from libcity.data.timeline import boundaries
from libcity.data.traffic_bundle import prepare_traffic


class ESGWindows(Dataset):
    def __init__(self,windows): self.windows=windows
    def __len__(self): return len(self.windows)
    def __getitem__(self,index):
        batch=self.windows[index]
        external=batch.pop('history_features',None)
        if external is not None:
            batch['X']=torch.cat((batch['X'],external[:,None,:].expand(-1,batch['X'].shape[1],-1)),dim=-1)
        return batch


def prepare_esg(data,training,evaluation,seed,logger=None):
    if data['sampling']['name']!='window' or data['graph']['name']!='dynamic':
        raise ValueError('ESG requires continuous windows and dynamic graphs')
    folder=Path(data['data_dir'])/data['dataset']
    raw=json.loads((folder/'config.json').read_text(encoding='utf-8'));raw={**raw,**raw.get('info',{})}
    geo=pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    grid={'row_id','column_id'}<=set(geo)
    plain=dict(data,graph=dict(name='none',symmetric=False,self_loops=False))
    if grid: plain['sampling']=dict(name='grid_window')
    bundle=(prepare_grid if grid else prepare_traffic)(plain,training,evaluation,seed,logger)
    meta=bundle.metadata;dataset=bundle.train.dataset
    history=dataset.arrays['history'] if grid else dataset.history
    clock=dataset.arrays['clock'] if grid else dataset.clock
    if np.isinf(history).any(): raise ValueError('ESG historical observations must not contain infinities')
    cleaned=np.nan_to_num(history,nan=0.)
    end=boundaries(len(clock),data['split'],clock)[0]
    limit=data['graph']['history_steps']
    stop=min(end,limit) if limit else end
    channels=data['graph']['history_channels']
    channels=list(range(meta.output_dim)) if channels is None else channels
    if not set(channels)<=set(range(meta.output_dim)):
        raise ValueError('ESG history_channels must select leading target observations, not calendar or future features')
    interval=np.timedelta64(meta.info.get('time_intervals',300),'s')
    if np.any(np.diff(clock[:stop])!=interval):
        raise ValueError('ESG graph history has timestamp gaps; choose history_steps within the initial contiguous training segment')
    values=bundle.scaler.transform(cleaned[:stop,:,channels])
    # Preserve LibCity's time-then-target-channel interleaving for its Conv1d.
    values=values.transpose(0,2,1).reshape(-1,meta.num_nodes).astype(np.float32)
    if not np.isfinite(values).all(): raise ValueError('ESG normalized graph history must be finite')
    external=meta.input_layout['historical_external_dim'] if grid else 0
    description=dict(source='original_training_prefix',start_index=0,end_index_exclusive=stop,
        requested_history_steps=limit,channels=channels,rows=len(values),first_timestamp=str(clock[0]),
        last_timestamp=str(clock[stop-1]),scalar_order='timestamp_then_selected_target_channel',
        normalization='target_scaler',missing_observations='raw_zero_before_scaling')
    layout=dict(meta.input_layout,kind='window',graph_history=description,
        source_layout='grid_window' if grid else 'point_window',input_missing_observations='raw_zero_before_scaling')
    if grid:
        layout.update(observed_feature_indices=list(range(meta.feature_dim)),
            global_history_features='broadcast_into_X',historical_external_dim=0)
    bundle.metadata=replace(meta,feature_dim=meta.feature_dim+external,input_layout=layout,
        static_data={**meta.static_data,'graph_history':values})
    for name in ('train','val','test'):
        loader=getattr(bundle,name)
        # GridWindows owns one raw array for input and target slicing and already
        # imputes only its input copy. Replacing that array would erase y_raw NaNs.
        # Point windows keep history and targets separately, so only history changes.
        if not grid: loader.dataset.history=cleaned
        setattr(bundle,name,DataLoader(ESGWindows(loader.dataset),batch_size=loader.batch_size,num_workers=loader.num_workers,
            shuffle=training['shuffle'] if name=='train' else False,generator=loader.generator))
    bundle.split_summary.update(adapter='esg',graph_history=description,input_missing_observations='raw_zero_before_scaling')
    if logger: logger.info('ESG observed training graph history: %s',description)
    return bundle
