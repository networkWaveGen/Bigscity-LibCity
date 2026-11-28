"""STDN's observed volume/flow contexts and complete native training lifecycle."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config,merge_config
from libcity.data.bundle import DatasetMetadata
from libcity.data.stdn_bundle import prepare_stdn
from libcity.data.sources import atomic_sources
from libcity.executor.evaluate import ForecastLoss
from libcity.executor.train import train_loop
from libcity.model.traffic_flow_prediction.runtime_stdn import STDN
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import stdn_fixture
from runtime_helpers import make_context

pytestmark=pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(nodes,features,outputs,steps,periods,length,radius,vector):
    return DatasetMetadata('tiny',nodes,features,outputs,input_layout=dict(kind='stdn_grid',length=steps,
        periods=periods,period_steps=length,cnn_nbhd_size=radius,feature_vector_width=vector))


@pytest.mark.parametrize('case',[(2,6,2,2,3,2,3,1,8,6,7),(1,1,3,1,1,1,1,0,3,4,5),
    (2,3,1,1,2,1,5,2,7,3,4),(1,1,2,2,7,3,3,3,160,128,128)],ids=['rectangular','singletons','long-period','defaults'])
def test_original_flow_gates_hierarchical_attention_and_all_parameter_input_gradients(case):
    batch,nodes,features,outputs,steps,periods,length,radius,vector,flat,hidden=case
    torch.set_num_threads(1);torch.manual_seed(17)
    meta=metadata(nodes,features,outputs,steps,periods,length,radius,vector)
    current=STDN(meta,steps,1,flat,hidden).double()
    legacy=importlib.import_module('libcity.model.traffic_flow_prediction.STDN').STDN
    old=legacy(dict(input_window=steps,output_window=1,att_lstm_num=periods,att_lstm_seq_len=length,
        cnn_nbhd_size=radius,cnn_flat_size=flat,lstm_out_size=hidden,device=torch.device('cpu')),
        dict(feature_vec_len=vector,nbhd_type=features,output_dim=outputs,len_row=1,len_column=nodes)).double()
    old.load_state_dict(current.state_dict())
    patch=2*radius+1;sampled=steps+periods*length
    x=torch.rand(batch,sampled,nodes,patch,patch,features,dtype=torch.float64,requires_grad=True)
    context=torch.rand(batch,sampled,nodes,4*patch**2+vector,dtype=torch.float64,requires_grad=True)
    other_x=x.detach().clone().requires_grad_();other_context=context.detach().clone().requires_grad_()
    flow=other_context[...,:4*patch**2].reshape(batch,sampled,nodes,patch,patch,4)
    vec=other_context[...,4*patch**2:]
    def packed(values,left,right,tail):
        return values[:,left:right].transpose(1,2).reshape(batch,1,1,nodes,*tail)
    reference=dict(nbhd_inputs=packed(other_x,0,steps,(steps,patch,patch,features)),
        flow_inputs=packed(flow,0,steps,(steps,patch,patch,4)),lstm_inputs=packed(vec,0,steps,(steps,vector)),
        flatten_att_nbhd_inputs=packed(other_x,steps,sampled,(periods*length,patch,patch,features)),
        flatten_att_flow_inputs=packed(flow,steps,sampled,(periods*length,patch,patch,4)),
        att_lstm_inputs=packed(vec,steps,sampled,(periods,length,vector)))
    expected=old(reference).reshape(batch,1,nodes,outputs)
    actual=current(dict(X=x,history_features=context))
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
    coefficient=torch.linspace(.2,1.2,actual.numel(),dtype=actual.dtype).reshape_as(actual)
    (actual*coefficient).sum().backward();(expected*coefficient).sum().backward()
    for name,parameter in current.named_parameters():
        assert parameter.grad is not None,name
        torch.testing.assert_close(parameter.grad,dict(old.named_parameters())[name].grad,rtol=1e-8,atol=1e-10)
    torch.testing.assert_close(x.grad,other_x.grad,rtol=1e-8,atol=1e-10)
    torch.testing.assert_close(context.grad,other_context.grad,rtol=1e-8,atol=1e-10)


def configuration(root,name='run'):
    return dict(model=dict(name='STDN',params=dict(cnn_flat_size=8,lstm_out_size=7)),
        data=dict(dataset='STDNTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),input_window=3,output_window=2,pad_training=False,
            sampling=dict(att_lstm_num=2,att_lstm_seq_len=3,hist_feature_daynum=1,last_feature_num=3,cnn_nbhd_size=1,nbhd_size=1)),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
            scheduler=dict(name='plateau',interval='metric',monitor='val_loss',factor=.5,patience=0)),
        evaluation=dict(batch_size=4),runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def bundle(cfg): return prepare_stdn(cfg['data'],cfg['training'],cfg['evaluation'],17)


def reference_sample(dataset,start,positions,shape):
    volume,flows=dataset.arrays['volume'],dataset.arrays['flow'];settings=dataset.settings
    nodes,features=volume.shape[1:]
    patches=[];contexts=[]
    for offset in dataset.offsets:
        time=start+int(offset);step_patches=[];step_contexts=[]
        for node,position in enumerate(positions):
            row,column=divmod(position,shape[1]);patches_for_node=[]
            for radius in (settings['cnn_nbhd_size'],settings['nbhd_size']):
                patch=np.zeros((2*radius+1,2*radius+1,features));flow=np.zeros((2*radius+1,2*radius+1,4))
                for dr in range(-radius,radius+1):
                    for dc in range(-radius,radius+1):
                        target=(row+dr)*shape[1]+column+dc
                        if 0<=row+dr<shape[0] and 0<=column+dc<shape[1]:
                            other=list(positions).index(target)
                            patch[dr+radius,dc+radius]=np.nan_to_num(volume[time,other],nan=0.)
                            flow[dr+radius,dc+radius]=np.nan_to_num([flows[time,node,other,0],flows[time,other,node,0],
                                flows[time-1,node,other,1],flows[time-1,other,node,1]],nan=0.)
                patches_for_node.append((patch,flow))
            hist=[volume[time-day*dataset.day,node] for day in range(settings['hist_feature_daynum'],0,-1)]
            last=[volume[slot,node] for slot in range(time-settings['last_feature_num'],time)]
            vector=np.concatenate((np.asarray(hist).ravel(),np.asarray(last).ravel(),patches_for_node[1][0].ravel()))
            step_patches.append(dataset.scaler.transform(patches_for_node[0][0]))
            step_contexts.append(np.r_[dataset.flow_scaler.transform(patches_for_node[0][1]).ravel(),dataset.scaler.transform(vector)])
        patches.append(step_patches);contexts.append(step_contexts)
    return np.array(patches),np.array(contexts)


@pytest.mark.parametrize('scaler',['normal','standard'])
def test_causal_patches_four_flow_directions_and_daily_recent_local_vectors_match_independent_loops(tmp_path,scaler):
    folder=stdn_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path);cfg['data'].update(scaler=dict(name=scaler),features=dict(flow_scaler=dict(name=scaler)))
    cfg=load_config(config=cfg);data=bundle(cfg);dataset=data.val.dataset
    assert data.metadata.num_nodes==6 and data.metadata.feature_dim==2
    assert data.metadata.input_layout['history_offsets']==(-3,-2,-1,-49,-48,-47,-25,-24,-23)
    sample=dataset[0];start=int(dataset.target_starts[0]);assert start==96
    expected_x,expected_context=reference_sample(dataset,start,data.metadata.input_layout['node_positions'],(2,3))
    np.testing.assert_allclose(sample['X'],expected_x,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(sample['history_features'],expected_context,rtol=1e-6,atol=1e-7)
    assert sample['X'].shape==(9,6,3,3,2) and sample['history_features'].shape==(9,6,62)
    # Raw CSV coordinates identify each loaded tensor node independently of row order.
    geo=pd.read_csv(folder/'STDNTiny.geo');grid=pd.read_csv(folder/'STDNTiny.grid')
    row=grid[(grid['time']==sorted(grid['time'].unique())[start])&(grid['row_id']==geo.iloc[0]['row_id'])&(grid['column_id']==geo.iloc[0]['column_id'])].iloc[0]
    np.testing.assert_allclose(sample['y_raw'][0,0],row[['value_0','value_1']].to_numpy(dtype=float),rtol=1e-6)
    targets=[{int(t+h) for t in loader.dataset.target_starts for h in range(2)} for loader in (data.train,data.val,data.test)]
    assert not targets[0]&targets[1] and not targets[0]&targets[2] and not targets[1]&targets[2]
    assert int(data.test.dataset.target_starts[-1])==158
    assert not data.metadata.input_layout['node_positions'].__class__ is list
    sources=atomic_sources(cfg['data'])
    assert any(row['name']=='STDNTiny.gridod' and row['size'] for row in sources)
    assert not any(row['name']=='STDNTiny.dyna' and row['size'] is None for row in sources)


def test_future_volumes_flows_do_not_change_inputs_or_train_scalers(tmp_path):
    folder=stdn_fixture(tmp_path/'raw');cfg=load_config(config=configuration(tmp_path))
    first=bundle(cfg);expected=first.val.dataset[0]
    stop=96;np.testing.assert_allclose(first.scaler.max,np.nanmax(first.train.dataset.arrays['volume'][:stop]))
    maximum=first.metadata.input_layout['flow_scaler']['state']['max']
    assert maximum==pytest.approx(np.nanmax(first.train.dataset.arrays['flow'][:stop]))
    for suffix,columns in (('.grid',['value_0','value_1']),('.gridod',['same_slot','next_slot'])):
        path=folder/('STDNTiny'+suffix);frame=pd.read_csv(path);future=frame['time']>=sorted(frame['time'].unique())[stop]
        frame.loc[future,columns]+=10000;frame.to_csv(path,index=False)
    changed=bundle(cfg)
    assert vars(first.scaler)==vars(changed.scaler)
    assert first.metadata.input_layout['flow_scaler']==changed.metadata.input_layout['flow_scaler']
    for name in ('X','history_features'): torch.testing.assert_close(expected[name],changed.val.dataset[0][name],rtol=0,atol=0)
    assert not torch.equal(expected['y_raw'],changed.val.dataset[0]['y_raw'])


def test_missing_history_is_imputed_without_changing_raw_zero_or_nan_targets(tmp_path):
    folder=stdn_fixture(tmp_path/'raw');path=folder/'STDNTiny.grid';frame=pd.read_csv(path)
    times=sorted(frame['time'].unique())
    frame.loc[frame['time'].isin([times[95],times[96]])&(frame['row_id']==0)&(frame['column_id']==0),'value_0']=float('nan')
    frame.loc[(frame['time']==times[96])&(frame['row_id']==0)&(frame['column_id']==1),'value_0']=0.
    frame.to_csv(path,index=False)
    cfg=configuration(tmp_path);cfg['data']['scaler']=dict(name='standard');cfg=load_config(config=cfg)
    result=bundle(cfg);sample=result.val.dataset[0];positions=list(result.metadata.input_layout['node_positions'])
    assert torch.isnan(sample['y_raw'][0,positions.index(0),0])
    assert sample['y_raw'][0,positions.index(1),0]==0 and sample['y'][0,positions.index(1),0]!=0
    assert torch.isfinite(sample['X']).all() and torch.isfinite(sample['history_features']).all()
    loss=ForecastLoss(cfg['training']['loss'],result.scaler,2);prediction=torch.zeros_like(sample['y'])
    outcome=loss(prediction,sample['y'],sample['y_raw'])
    valid=torch.isfinite(sample['y_raw'])&(sample['y_raw']!=0)
    expected=(result.scaler.inverse_transform(prediction)[valid]-sample['y_raw'][valid]).square().mean()
    assert outcome.weight==int(valid.sum());torch.testing.assert_close(outcome.loss,expected)


def test_geo_row_permutation_keeps_volume_flow_patches_on_the_same_physical_nodes(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(17)
    folder=stdn_fixture(tmp_path/'raw');cfg=load_config(config=configuration(tmp_path))
    original=bundle(cfg);first=STDN(original.metadata,3,2,8,7)
    path=folder/'STDNTiny.geo';geo=pd.read_csv(path);geo.iloc[[4,2,5,0,3,1]].to_csv(path,index=False)
    changed=bundle(cfg);second=STDN(changed.metadata,3,2,8,7);second.load_state_dict(first.state_dict())
    ids=list(original.metadata.info['node_ids']);order=[ids.index(value) for value in changed.metadata.info['node_ids']]
    a={name:value.unsqueeze(0) for name,value in original.val.dataset[0].items()}
    b={name:value.unsqueeze(0) for name,value in changed.val.dataset[0].items()}
    torch.testing.assert_close(a['X'][:,:,order],b['X'],rtol=0,atol=0)
    torch.testing.assert_close(a['history_features'][:,:,order],b['history_features'],rtol=0,atol=0)
    torch.testing.assert_close(first(a)[:,:,order],second(b),rtol=2e-6,atol=1e-7)


def test_raw_cache_reuse_across_windows_patch_radius_scalers_and_no_full_atom_reads(tmp_path,monkeypatch):
    stdn_fixture(tmp_path/'raw');cfg=load_config(config=configuration(tmp_path));first=bundle(cfg)
    original=pd.read_csv
    def guarded(path,*args,**kwargs):
        if Path(path).suffix in ('.grid','.gridod','.ext') and kwargs.get('nrows')!=0: pytest.fail('cache hit re-read complete atoms')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',guarded)
    cfg['data'].update(input_window=2,output_window=3,scaler=dict(name='none'))
    cfg['data']['sampling'].update(cnn_nbhd_size=0,nbhd_size=0)
    cfg['data']['features']['flow_scaler']=dict(name='minmax01')
    second=bundle(cfg)
    assert second.split_summary['cache']['status']==second.split_summary['flow_cache']['status']=='hit'
    for key in ('cache','flow_cache'): assert second.split_summary[key]['path']==first.split_summary[key]['path']
    assert second.val.dataset[0]['X'].shape==(8,6,1,1,2)


@pytest.mark.parametrize('damage',['dtype','negative','clock'])
def test_bad_flow_cache_is_rebuilt_from_real_aligned_atoms(tmp_path,damage):
    stdn_fixture(tmp_path/'raw');cfg=load_config(config=configuration(tmp_path));first=bundle(cfg)
    path=first.split_summary['flow_cache']['path']
    with np.load(path,allow_pickle=False) as source: arrays={name:source[name] for name in source.files}
    if damage=='dtype': arrays['flow']=arrays['flow'].astype(str)
    elif damage=='negative': arrays['flow'][0,0,0,0]=-1
    else: arrays['clock'][0]+=np.timedelta64(1,'m')
    np.savez_compressed(path,**arrays)
    assert bundle(cfg).split_summary['flow_cache']['status']=='miss'


@pytest.mark.parametrize('problem',['duplicate','missing','unknown-cell','clock','negative','columns','frequency'])
def test_invalid_flow_sources_or_daily_interval_are_rejected(tmp_path,problem):
    folder=stdn_fixture(tmp_path/'raw');cfg=load_config(config=configuration(tmp_path))
    path=folder/'STDNTiny.gridod';frame=pd.read_csv(path)
    if problem=='duplicate': frame=pd.concat([frame,frame.iloc[:1]])
    elif problem=='missing': frame=frame.iloc[1:]
    elif problem=='unknown-cell': frame.loc[0,'origin_row_id']=999
    elif problem=='clock': frame['time']=pd.to_datetime(frame['time'],utc=True).add(pd.Timedelta(minutes=1)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    elif problem=='negative': frame.loc[0,'next_slot']=-1
    elif problem=='columns':
        p=folder/'config.json';raw=json.loads(p.read_text());raw['flow_col']=['same_slot'];p.write_text(json.dumps(raw))
    else:
        p=folder/'config.json';raw=json.loads(p.read_text());raw['time_intervals']=777;p.write_text(json.dumps(raw))
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='STDN'): bundle(cfg)


@pytest.mark.parametrize('boundary',['files','gap'])
def test_history_features_and_targets_never_cross_file_or_clock_boundaries(tmp_path,boundary):
    folder=stdn_fixture(tmp_path/'raw',steps=320);cfg=load_config(config=configuration(tmp_path))
    for suffix in ('.grid','.gridod'):
        frame=pd.read_csv(folder/('STDNTiny'+suffix));times=sorted(frame['time'].unique())
        if boundary=='files':
            for index,part in enumerate((times[:160],times[160:])): frame[frame['time'].isin(part)].to_csv(folder/(f'part{index}'+suffix),index=False)
        else:
            selection=frame['time'].isin(times[150:])
            frame.loc[selection,'time']=(pd.to_datetime(frame.loc[selection,'time'],utc=True)+pd.Timedelta(hours=1)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            frame.to_csv(folder/('STDNTiny'+suffix),index=False)
    if boundary=='files':
        path=folder/'config.json';raw=json.loads(path.read_text());raw.update(data_files=['part0','part1'],flow_files=['part0','part1']);path.write_text(json.dumps(raw))
    result=bundle(cfg)
    for loader in (result.train,result.val,result.test):
        for start in loader.dataset.target_starts:
            left,right=int(start)-73,int(start)+1
            assert not (left<(160 if boundary=='files' else 150)<=right)


def equal(left,right):
    if isinstance(left,torch.Tensor): torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True): equal(a,b)
    else: assert left==right


@pytest.mark.parametrize('scaler',['normal','standard'])
def test_actual_native_train_independent_raw_evaluate_all_horizons_and_exact_resume(tmp_path,scaler):
    stdn_fixture(tmp_path/'raw')
    def config(name):
        cfg=configuration(tmp_path,name);cfg['data'].update(scaler=dict(name=scaler),features=dict(flow_scaler=dict(name=scaler)));return cfg
    full=run_experiment('train',config=config('full'))
    evaluated=run_experiment('evaluate',config=config('eval'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        prediction,truth=archive['prediction'],archive['truth'];valid=np.isfinite(truth)&(truth!=0)
        assert prediction.shape==(31,2,6,2)
        assert evaluated.metrics['loss']==pytest.approx(np.square(prediction[valid]-truth[valid]).mean(),rel=2e-6,abs=1e-8)
    assert not (tmp_path/'runs/eval/checkpoints').exists() and not (tmp_path/'runs/eval/metrics.jsonl').exists()
    partial=make_context(config('partial'));before=partial.model.nbhd_cnns_1st[0][0].weight.detach().clone()
    try:
        # Model forwards remain batch/node independent and ignore targets.
        rng=torch.get_rng_state()
        batch=next(iter(partial.bundle.val));values=partial.loss_fn.inputs(batch)
        prediction=partial.model(values);separate=torch.cat([partial.model({name:value[i:i+1] for name,value in values.items()}) for i in range(len(batch['X']))])
        torch.testing.assert_close(prediction,separate,rtol=2e-6,atol=1e-7)
        torch.testing.assert_close(prediction,partial.model({**values,'y':torch.full_like(batch['y'],1e6)}),rtol=0,atol=0)
        prediction.retain_grad();partial.loss_fn(prediction,batch['y'],batch['y_raw']).loss.backward()
        assert all(prediction.grad[:,step].abs().sum()>0 for step in range(2))
        partial.model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        train_loop(partial,until_epoch=1)
        assert not torch.equal(before,partial.model.nbhd_cnns_1st[0][0].weight)
    finally: partial.tracker.close()
    resumed=run_experiment('train',config=config('resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual=[torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'): equal(expected[key],actual[key])


@pytest.mark.parametrize('singleton',[False,True])
def test_real_tail_cpu_amp_and_single_grid_time_period_without_extra_history(tmp_path,singleton):
    stdn_fixture(tmp_path/'raw',steps=96 if singleton else 160,rows=1 if singleton else 2,columns=1 if singleton else 3)
    cfg=configuration(tmp_path,'amp');cfg['training'].update(epochs=1,amp=True)
    if singleton:
        cfg['data'].update(input_window=1,sampling=dict(att_lstm_num=1,att_lstm_seq_len=1,hist_feature_daynum=0,
            last_feature_num=0,cnn_nbhd_size=0,nbhd_size=0))
    result=run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


def test_spawn_worker_and_changed_flow_source_resume_guard(tmp_path):
    folder=stdn_fixture(tmp_path/'raw');cfg=configuration(tmp_path,'spawn');cfg['training']['num_workers']=1
    context=make_context(cfg)
    try:
        batch=next(iter(context.bundle.train));assert batch['X'].shape==(5,9,6,3,3,2)
        train_loop(context,until_epoch=1)
    finally: context.tracker.close()
    path=folder/'STDNTiny.gridod';frame=pd.read_csv(path)
    changed=frame.index[frame['time']>=sorted(frame['time'].unique())[128]][0]
    frame.loc[changed,'same_slot']+=1;frame.to_csv(path,index=False)
    cfg['runtime']['exp_id']='wrong-source'
    with pytest.raises(ValueError,match='Atomic|source'): run_experiment('train',config=cfg,resume=context.directory/'checkpoints/last.pt')


@pytest.mark.parametrize('change',[
    dict(model=dict(params=dict(cnn_flat_size=0))),dict(model=dict(params=dict(lstm_out_size=0))),
    dict(data=dict(sampling=dict(att_lstm_num=0))),dict(data=dict(sampling=dict(att_lstm_seq_len=2))),
    dict(data=dict(sampling=dict(hist_feature_daynum=-1))),dict(data=dict(sampling=dict(last_feature_num=-1))),
    dict(data=dict(sampling=dict(cnn_nbhd_size=-1))),dict(data=dict(sampling=dict(nbhd_size=.5))),
    dict(data=dict(graph=dict(name='binary'))),dict(data=dict(graph=dict(self_loops=True))),
    dict(data=dict(features=dict(load_external=True))),dict(data=dict(features=dict(known_future_columns=['weather']))),
    dict(data=dict(features=dict(flow_scaler=dict(name='broken')))),dict(data=dict(sampling=dict(name='window')))])
def test_invalid_stdn_configuration_is_rejected_before_data_reads(change):
    with pytest.raises((ValueError,TypeError)):
        load_config(config=merge_config(dict(model=dict(name='STDN'),data=dict(dataset='missing')),change))


def test_default_protocol_explicitly_records_single_step_seven_recent_and_three_shifted_days():
    cfg=load_config(config=dict(model=dict(name='STDN'),data=dict(dataset='missing')))
    assert cfg['data']['input_window']==7 and cfg['data']['output_window']==1
    assert cfg['data']['sampling']==dict(name='stdn_grid',att_lstm_num=3,att_lstm_seq_len=3,hist_feature_daynum=7,
        last_feature_num=48,cnn_nbhd_size=3,nbhd_size=2)
