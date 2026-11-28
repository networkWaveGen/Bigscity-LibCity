"""Causal STDN volume/OD patches are sliced on demand from unexpanded atoms."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset

from libcity.data.bundle import DataBundle,DatasetMetadata
from libcity.data.grid_traffic import prepare_grid
from libcity.data.timeline import boundaries,partition_windows
from libcity.data.traffic_bundle import fit_scaler
from libcity.data.traffic_cache import prepare_arrays
from libcity.utils.runtime_checkpoint import plain


def neighborhood(positions,shape,radius):
    positions=np.asarray(positions,dtype=np.int64)
    rows,columns=np.divmod(positions,shape[1])
    delta=np.arange(-radius,radius+1)
    row=rows[:,None,None]+delta[None,:,None]
    column=columns[:,None,None]+delta[None,None,:]
    valid=(row>=0)&(row<shape[0])&(column>=0)&(column<shape[1])
    inverse=np.empty(len(positions),dtype=np.int64);inverse[positions]=np.arange(len(positions))
    index=inverse[np.clip(row,0,shape[0]-1)*shape[1]+np.clip(column,0,shape[1]-1)]
    return index,valid


class STDNWindows(Dataset):
    def __init__(self,arrays,starts,offsets,history,horizon,outputs,scaler,flow_scaler,settings,shape,positions,day):
        self.arrays,self.target_starts,self.offsets=arrays,starts,offsets
        self.history,self.horizon,self.outputs=history,horizon,outputs
        self.scaler,self.flow_scaler=scaler,flow_scaler
        self.settings,self.day=settings,day
        self.patch,self.valid=neighborhood(positions,shape,settings['cnn_nbhd_size'])
        self.feature_patch,self.feature_valid=neighborhood(positions,shape,settings['nbhd_size'])

    def __len__(self): return len(self.target_starts)

    def __getitem__(self,index):
        start=int(self.target_starts[index]);times=start+self.offsets
        volume,flow=self.arrays['volume'],self.arrays['flow']
        patch=volume[times[:,None,None,None],self.patch[None,...]]
        patch=np.where(self.valid[None,...,None],np.nan_to_num(patch,nan=0.),0.)
        center=np.arange(volume.shape[1])[None,:,None,None];neighbor=self.patch[None,...]
        t=times[:,None,None,None]
        flows=np.stack((flow[t,center,neighbor,0],flow[t,neighbor,center,0],
            flow[t-1,center,neighbor,1],flow[t-1,neighbor,center,1]),axis=-1)
        flows=np.where(self.valid[None,...,None],np.nan_to_num(flows,nan=0.),0.)
        historic=np.arange(self.settings['hist_feature_daynum'],0,-1)*self.day
        recent=np.arange(self.settings['last_feature_num'],0,-1)
        features=[]
        for offsets in (historic,recent):
            values=volume[times[:,None]-offsets[None,:]].transpose(0,2,1,3)
            features.append(np.nan_to_num(values,nan=0.).reshape(len(times),len(center[0]),-1))
        local=volume[times[:,None,None,None],self.feature_patch[None,...]]
        local=np.where(self.feature_valid[None,...,None],np.nan_to_num(local,nan=0.),0.)
        features.append(local.reshape(len(times),volume.shape[1],-1))
        vector=self.scaler.transform(np.concatenate(features,axis=-1))
        context=np.concatenate((self.flow_scaler.transform(flows).reshape(len(times),volume.shape[1],-1),vector),axis=-1)
        raw=volume[start:start+self.horizon,:,:self.outputs].copy()
        return {name:torch.tensor(value,dtype=torch.float32) for name,value in dict(
            X=self.scaler.transform(patch),history_features=context,y=self.scaler.transform(raw),y_raw=raw).items()}


def prepare_stdn(data,training,evaluation,seed,logger=None):
    sampling=data['sampling']
    if sampling['name']!='stdn_grid' or data['graph']['name']!='none':
        raise ValueError('STDN requires stdn_grid sampling without a static graph')
    folder=Path(data['data_dir'])/data['dataset']
    raw=json.loads((folder/'config.json').read_text(encoding='utf-8'));raw={**raw,**raw.get('info',{})}
    interval=raw.get('time_intervals',300)
    if type(interval) is not int or interval<1 or 86400%interval:
        raise ValueError('STDN sampling interval must divide one day exactly')
    day=86400//interval
    options=dict(load_external=False,time_in_day=False,day_in_week=False,known_future_columns=[])
    if any(data['features'][name] for name in options):
        raise ValueError('STDN uses observed volume/flow context; external and known-future switches must be disabled')
    # Reuse the validated volume reader/cache; discard its cheap generic windows.
    source=prepare_grid(dict(data,sampling=dict(name='grid_window'),features=options),training,evaluation,seed,logger)
    volume_arrays=source.train.dataset.arrays
    nodes=source.metadata.num_nodes
    positions=np.asarray(source.metadata.input_layout['node_positions'])
    shape=tuple(source.metadata.input_layout['grid_shape'])
    count,length=sampling['att_lstm_num'],sampling['att_lstm_seq_len']
    if length%2!=1 or length//2>=day:
        raise ValueError('STDN periodic window must be odd and remain before the forecast origin')
    period=np.arange(-count*day,0,day)[:,None]+np.arange(-(length//2),length//2+1)[None,:]
    offsets=np.r_[np.arange(-data['input_window'],0),period.ravel()]
    lookback=int(-offsets.min()+max(sampling['hist_feature_daynum']*day,sampling['last_feature_num'],1))
    names=raw.get('flow_files',raw.get('data_files',data['dataset']))
    names=[names] if isinstance(names,str) else names
    if not isinstance(names,list) or not names or any(not isinstance(name,str) or not name for name in names):
        raise ValueError('STDN flow_files must contain nonempty gridod basenames')
    paths=[folder/(name+'.gridod') for name in names]
    reserved={'dyna_id','type','time','origin_id','destination_id','origin_row_id','origin_column_id',
        'destination_row_id','destination_column_id'}
    columns=raw.get('flow_col') or [name for name in pd.read_csv(paths[0],nrows=0) if name not in reserved]
    if not isinstance(columns,list) or len(columns)!=2 or len(set(columns))!=2 or any(not isinstance(name,str) or not name for name in columns):
        raise ValueError('STDN flow_col must select exactly two distinct gridod flow columns')
    ids=list(source.metadata.info['node_ids']);id_index={value:index for index,value in enumerate(ids)}
    cell_index={divmod(int(value),shape[1]):index for index,value in enumerate(positions)}
    def build():
        values,clocks,intervals=[],[],[];offset=0
        for path in paths:
            frame=pd.read_csv(path)
            if 'time' not in frame or not set(columns)<=set(frame):
                raise ValueError('STDN gridod files require time and both flow columns')
            stamp=pd.to_datetime(frame['time'],utc=True,errors='raise')
            if stamp.isna().any(): raise ValueError('STDN flow timestamps must be finite')
            clock=pd.DatetimeIndex(stamp.unique()).sort_values()
            try:
                if {'origin_id','destination_id'}<=set(frame):
                    origins=np.array([id_index[value] for value in frame['origin_id']])
                    destinations=np.array([id_index[value] for value in frame['destination_id']])
                else:
                    origins=np.array([cell_index[(row,column)] for row,column in zip(frame['origin_row_id'],frame['origin_column_id'])])
                    destinations=np.array([cell_index[(row,column)] for row,column in zip(frame['destination_row_id'],frame['destination_column_id'])])
            except KeyError as error:
                raise ValueError('STDN flow endpoint is absent from grid .geo') from error
            indices=clock.get_indexer(stamp)
            pairs=(indices*nodes+origins)*nodes+destinations
            if not len(clock) or len(pairs)!=len(clock)*nodes**2 or len(np.unique(pairs))!=len(pairs):
                raise ValueError('STDN requires exactly one row per time/origin/destination; encode zero flows explicitly')
            array=np.empty((len(clock),nodes,nodes,2),dtype=np.float64)
            array[indices,origins,destinations]=frame[columns].to_numpy(dtype=np.float64)
            values.append(array);clocks.append(clock.to_numpy(dtype='datetime64[ns]'))
            intervals.append((offset,offset+len(clock)));offset+=len(clock)
        return dict(flow=np.concatenate(values),clock=np.concatenate(clocks),intervals=np.array(intervals,dtype=np.int64))
    def validate(arrays):
        flow,clock,intervals=(arrays[name] for name in ('flow','clock','intervals'))
        if (flow.dtype.kind not in 'fiu' or flow.shape!=(len(volume_arrays['clock']),nodes,nodes,2) or
                np.isinf(flow).any() or np.any(flow[np.isfinite(flow)]<0) or clock.dtype.kind!='M' or
                not np.array_equal(clock,volume_arrays['clock']) or not np.array_equal(intervals,volume_arrays['intervals'])):
            raise ValueError('STDN flow values, clocks and file boundaries must align with volume observations')
    flow_data=dict(data,features={},graph=dict(name='none'))
    flows,flow_cache=prepare_arrays(flow_data,raw,build,logger,namespace='stdn-flows-v1',
        source_paths=[folder/'config.json',folder/(raw.get('geo_file',data['dataset'])+'.geo'),*paths],
        fields=('flow','clock','intervals'),validate=validate)
    validate(flows)
    arrays=dict(volume=volume_arrays['history'],flow=flows['flow'],clock=volume_arrays['clock'],intervals=volume_arrays['intervals'])
    clock=arrays['clock'];cuts=boundaries(len(clock),data['split'],clock)
    flow_scaler=fit_scaler(data['features']['flow_scaler']['name'],arrays['flow'][:cuts[0]])
    horizon=data['output_window']
    starts=np.concatenate([np.arange(left+lookback,right-horizon+1) for left,right in arrays['intervals']])
    gaps=np.r_[False,np.diff(clock)!=np.timedelta64(source.metadata.info['time_intervals'],'s')].cumsum()
    starts=starts[gaps[starts-lookback]==gaps[starts+horizon-1]]
    partitions=partition_windows(starts,horizon,cuts,len(clock))
    feature_width=(sampling['hist_feature_daynum']+sampling['last_feature_num']+(2*sampling['nbhd_size']+1)**2)*source.metadata.feature_dim
    layout=dict(kind='stdn_grid',length=data['input_window'],history_offsets=offsets.tolist(),
        recent_steps=data['input_window'],periods=count,period_steps=length,grid_shape=shape,node_positions=positions.tolist(),
        cnn_nbhd_size=sampling['cnn_nbhd_size'],nbhd_size=sampling['nbhd_size'],feature_vector_width=feature_width,
        sampling=dict(sampling),flow_channels=4,flow_columns=columns,flow_feature_order=['current_out','current_in','previous_lag_out','previous_lag_in'],
        flow_scaler=dict(name=type(flow_scaler).__name__,state=plain(vars(flow_scaler))),
        flow_scaler_fit_end_exclusive=str(clock[cuts[0]]),target_semantics='raw_and_scaled',
        input_axes=['sampled_history','geo_node','patch_row','patch_column','observation'],
        history_features='flattened_flow_patch_then_daily_recent_local_volume',split_semantics='target_time_v1')
    meta=DatasetMetadata(source.metadata.name,nodes,source.metadata.feature_dim,source.metadata.output_dim,
        info=dict(source.metadata.info,flow_columns=columns),input_layout=layout)
    generator=torch.Generator().manual_seed(seed);loaders=[];splits={}
    for index,(name,selection) in enumerate(zip(('train','val','test'),partitions)):
        chosen=starts[selection]
        if not len(chosen): raise ValueError('STDN '+name+' has no complete causal windows; increase timeline or reduce sampled history')
        splits[name]=dict(samples=len(chosen),target_index_start=int(chosen.min()),target_index_end=int(chosen.max()+horizon-1))
        if index==0 and data['pad_training']: chosen=np.pad(chosen,(0,(-len(chosen))%training['batch_size']),mode='edge')
        dataset=STDNWindows(arrays,chosen,offsets,data['input_window'],horizon,meta.output_dim,source.scaler,flow_scaler,sampling,shape,positions,day)
        config=training if index==0 else evaluation
        loaders.append(DataLoader(dataset,batch_size=config['batch_size'],num_workers=config['num_workers'],
            shuffle=training['shuffle'] if index==0 else False,generator=generator if index==0 else None))
    summary=dict(semantics='target_time_v1',adapter='stdn_grid',input_window=data['input_window'],history_span_points=lookback,
        requested_input_window=data['input_window'],sampled_history_steps=len(offsets),offsets=offsets.tolist(),grid_shape=list(shape),
        intervals=splits,train=len(loaders[0].dataset),val=len(loaders[1].dataset),test=len(loaders[2].dataset),
        train_end=str(clock[cuts[0]]),val_end=str(clock[cuts[1]]),scaler_fit_end_exclusive=str(clock[cuts[0]]),
        flow_scaler_fit_end_exclusive=str(clock[cuts[0]]),train_padded=data['pad_training'],cache=source.split_summary['cache'],flow_cache=flow_cache,
        sampling_gap_policy='whole_volume_flow_context_and_target_span_contiguous',input_missing_observations='zero_before_scaling')
    if logger: logger.info('STDN target-time split and flow alignment: %s',summary)
    return DataBundle(*loaders,meta,source.scaler,generator,summary)
