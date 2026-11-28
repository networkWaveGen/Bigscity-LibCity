"""D2STGNN math, unexpanded graph kernels, explicit curriculum and lifecycle."""
from copy import deepcopy
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from libcity.config.runtime import load_config,merge_config
from libcity.data.bundle import DatasetMetadata
from libcity.executor.d2stgnn_step import curriculum_step,train
from libcity.executor.evaluate import ForecastLoss
from libcity.executor.train import TrainState
from libcity.model.traffic_speed_prediction.runtime_d2stgnn import D2STGNN,STLocalizedConv,graph_orders,transition_supports
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,d2stgnn_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark=pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')
SMALL=dict(num_hidden=4,node_dim=3,forecast_dim=6,output_hidden=8,k_s=2,k_t=2,num_layers=2,time_emb_dim=3,
    gap=2,use_pre=False,dy_graph=True,sta_graph=True,use_time_of_day=True,use_day_of_week=True,
    attention_heads=4,dropout=0.,steps_per_day=0,dynamic_channel=0)


def metadata(nodes=4,numeric=2,outputs=2,time=True,week=True,interval=300):
    graph=np.zeros((nodes,nodes),dtype=np.float32)
    for index in range(nodes-1): graph[index,index+1]=index+1
    tid=numeric if time else None;dow=list(range(numeric+int(time),numeric+int(time)+7)) if week else []
    return DatasetMetadata('tiny',nodes,numeric+int(time)+7*int(week),outputs,adjacency=graph,
        info=dict(time_intervals=interval),input_layout=dict(kind='window',calendar=dict(time_in_day=tid,day_of_week=dow)))


def inputs(meta,batch,steps):
    x=torch.randn(batch,steps,meta.num_nodes,meta.feature_dim,dtype=torch.float64)
    calendar=meta.input_layout['calendar']
    if calendar['time_in_day'] is not None:
        x[...,calendar['time_in_day']]=(torch.arange(steps)%4)[None,:,None]/4
    if calendar['day_of_week']:
        days=F.one_hot(torch.arange(steps)%7,7).double()
        x[...,calendar['day_of_week']]=days[None,:,None]
    return x.requires_grad_()


PROFILES=[(6,5,2,dict(use_pre=True)),(5,4,2,dict(dy_graph=False)),(5,3,1,dict(dy_graph=False,sta_graph=False,use_pre=True)),
    (5,3,1,dict(dy_graph=False,sta_graph=False,use_time_of_day=False,use_day_of_week=False)),
    (5,4,2,dict(use_time_of_day=True,use_day_of_week=False)),(5,4,2,dict(use_time_of_day=False,use_day_of_week=True)),
    (12,12,1,dict(num_hidden=32,node_dim=10,forecast_dim=256,output_hidden=512,k_s=2,k_t=3,num_layers=5,time_emb_dim=10,gap=3))]


@pytest.mark.parametrize('history,horizon,outputs,changes',PROFILES,
    ids=['all-graphs-nondivisible','static','predefined-short-horizon','self-only','time-only','week-only','defaults'])
def test_original_full_predictions_all_live_gradients_and_batchnorm_state(history,horizon,outputs,changes):
    torch.set_num_threads(1);torch.manual_seed(17);params={**SMALL,**changes}
    meta=metadata(nodes=3,numeric=2,outputs=outputs,time=params['use_time_of_day'],week=params['use_day_of_week'])
    current=D2STGNN(meta,history,horizon,**params).double()
    effective_gap=min(horizon,params['gap'])
    config=dict(params,input_window=history,output_window=int(np.ceil(horizon/effective_gap))*effective_gap,
        node_hidden=params['node_dim'],add_time_in_day=params['use_time_of_day'],add_day_in_week=params['use_day_of_week'],device='cpu')
    config['gap']=effective_gap
    legacy=importlib.import_module('libcity.model.traffic_speed_prediction.D2STGNN')
    old=legacy.D2STGNN(config,dict(adj_mx=meta.adjacency,num_nodes=meta.num_nodes,feature_dim=meta.feature_dim,
        output_dim=outputs,num_batches=1)).double()
    # Non-buffer legacy graph tensors do not follow .double().
    if params['dy_graph']: old.dynamic_graph_constructor.mask.mask=[value.double() for value in old.dynamic_graph_constructor.mask.mask]
    for layer in old.layers:
        convolution=layer.dif_layer.localized_st_conv
        raw=[torch.tensor(np.asarray(legacy.transition_matrix(graph)).T).double() for graph in (meta.adjacency,meta.adjacency.T)]
        convolution.pre_defined_graph=convolution.get_graph(raw)
    state={key:value for key,value in current.state_dict().items() if key not in ('road_supports','predefined_orders','diagonal_mask') and not key.endswith('.pos_encoder.pe')}
    loaded=old.load_state_dict(state,strict=False);assert not loaded.unexpected_keys
    def inactive(name):
        return (name.endswith('.pos_encoder.pe') or '.localized_st_conv.bn.' in name or '.time_slot_embedding.' in name or
            name.startswith(f'layers.{params["num_layers"]-1}.inh_layer.backcast_fc.') or
            name.startswith(f'layers.{params["num_layers"]-1}.inh_layer.residual_decompose.') or
            (name=='T_i_D_emb' and not params['use_time_of_day']) or (name=='D_i_W_emb' and not params['use_day_of_week']))
    assert all(inactive(name) for name in loaded.missing_keys),loaded.missing_keys
    source=inputs(meta,2,history);other=source.detach().clone().requires_grad_()
    expected=old(dict(X=other))[:,:horizon];actual=current(dict(X=source))
    torch.testing.assert_close(actual,expected,rtol=2e-7,atol=2e-9)
    actual.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(source.grad,other.grad,rtol=2e-6,atol=1e-8)
    for name,value in current.named_parameters():
        assert value.grad is not None,name
        torch.testing.assert_close(value.grad,dict(old.named_parameters())[name].grad,rtol=3e-6,atol=2e-8)
    for name,value in current.named_buffers():
        if 'running_' in name or 'num_batches_tracked' in name:
            torch.testing.assert_close(value,old.state_dict()[name],rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('kernel',[1,2,3])
def test_unexpanded_localized_graph_matches_original_tiled_matmul_and_all_gradients(kernel):
    torch.manual_seed(17);current=STLocalizedConv(4,kernel,2,0.).double()
    legacy=importlib.import_module('libcity.model.traffic_speed_prediction.D2STGNN')
    reference=legacy.STLocalizedConv(4,[torch.eye(3,dtype=torch.float64)]*2,False,True,False,
        dict(k_s=1,k_t=kernel,dropout=0.,device='cpu')).double()
    reference.load_state_dict(current.state_dict(),strict=False)
    x=torch.randn(2,5,3,4,dtype=torch.float64,requires_grad=True);other=x.detach().clone().requires_grad_()
    graphs=[torch.randn(3,3,dtype=torch.float64,requires_grad=True),torch.randn(2,3,3,dtype=torch.float64,requires_grad=True)]
    copies=[graph.detach().clone().requires_grad_() for graph in graphs]
    tiled=[graph.unsqueeze(-2).expand(*graph.shape[:-1],kernel,graph.shape[-1]).flatten(-2) for graph in copies]
    actual=current(x,graphs);expected=reference(other,tiled,[])
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    actual.square().sum().backward();expected.square().sum().backward()
    for a,b in [(x,other),*zip(graphs,copies),*zip(current.parameters(),[dict(reference.named_parameters())[name] for name,_ in current.named_parameters()])]:
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-10,atol=1e-11)


def test_transition_orientation_graph_orders_and_diagonal_removal_only_after_powers():
    graph=np.array([[1.,2.,0.],[0.,0.,3.],[4.,0.,0.]])
    actual=transition_supports(graph,3).double().numpy()
    for index,value in enumerate((graph,graph.T)):
        np.testing.assert_allclose(actual[index],(value/value.sum(1,keepdims=True)).T,rtol=1e-7)
    tensor=torch.tensor(graph,dtype=torch.float64);mask=~torch.eye(3,dtype=torch.bool)
    for index,value in enumerate(graph_orders(tensor,3,mask),1):
        expected=np.linalg.matrix_power(graph,index).copy();np.fill_diagonal(expected,0.)
        np.testing.assert_allclose(value.numpy(),expected)
    assert torch.count_nonzero(transition_supports(np.zeros((1,1)),1))==0


def configuration(root,name='run'):
    return dict(model=dict(name='D2STGNN',params={**SMALL,'attention_heads':2,'dropout':.2}),
        data=dict(dataset='DecoupleTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),input_window=6,output_window=3,pad_training=False),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,strategy=dict(cl_decay_epochs=1),
            scheduler=dict(name='plateau',interval='metric',monitor='val_loss',factor=.5,patience=0)),
        evaluation=dict(batch_size=4),runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def equal(left,right):
    if isinstance(left,torch.Tensor): torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True): equal(a,b)
    else: assert left==right


@pytest.mark.parametrize('scaler',['standard','minmax01'])
def test_actual_curriculum_dropout_bn_train_evaluate_and_exact_full_resume(tmp_path,scaler):
    d2stgnn_fixture(tmp_path/'raw')
    def config(name):
        cfg=configuration(tmp_path,name);cfg['data']['scaler']=dict(name=scaler);return cfg
    full=run_experiment('train',config=config('full'))
    evaluated=run_experiment('evaluate',config=config('evaluated'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as result:
        prediction,truth=result['prediction'],result['truth'];valid=np.isfinite(truth)&(truth!=0)
        assert prediction.shape==(10,3,4,2)
        assert evaluated.metrics['loss']==pytest.approx(np.abs(prediction[valid]-truth[valid]).mean(),rel=2e-6)
    assert not (tmp_path/'runs/evaluated/checkpoints').exists() and not (tmp_path/'runs/evaluated/metrics.jsonl').exists()
    partial=make_context(config('partial'))
    try: train(partial,until_epoch=1)
    finally: partial.tracker.close()
    resumed=run_experiment('train',config=config('resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual=[torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'): equal(expected[key],actual[key])
    assert actual['train_state']['task_state']['d2stgnn_curriculum_horizon']==3


def test_curriculum_is_epoch_based_counts_raw_valid_targets_and_ignores_unsupervised_horizons():
    loss=ForecastLoss(dict(name='masked_mae',mask_value=0.),NoneScaler(),1)
    model=nn.Module();model.prediction=nn.Parameter(torch.ones(2,3,2,1))
    model.forward=lambda batch:model.prediction
    raw=torch.zeros(2,3,2,1);raw[:,0,0]=2.;raw[:,1,0]=3.;raw[:,2,0]=4.;raw[0,0,1]=float('nan')
    batch=dict(X=torch.zeros(2,1,2,1),y=raw,y_raw=raw)
    strategy=dict(use_curriculum_learning=True,cl_decay_epochs=2);step=curriculum_step(loss,strategy)
    for epoch,horizon in [(0,1),(1,1),(2,2),(3,2),(4,3),(100,3)]:
        state=TrainState(epoch=epoch,batches_seen=99);model.zero_grad();actual=step(model,batch,state)
        assert actual.weight==2*horizon and state.task_state['d2stgnn_curriculum_horizon']==horizon
        expected=loss(model.prediction[:,:horizon],raw[:,:horizon],raw[:,:horizon])
        torch.testing.assert_close(actual.loss,expected.loss);actual.loss.backward()
        assert torch.count_nonzero(model.prediction.grad[:,horizon:])==0
    strategy['use_curriculum_learning']=False
    assert step(model,batch,TrainState()).weight==6


def test_eval_identity_batch_independence_static_ownership_and_calendar_numeric_columns():
    torch.set_num_threads(1);torch.manual_seed(17);meta=metadata(numeric=3,interval=3600)
    model=D2STGNN(meta,6,5,**SMALL).double().eval()
    assert model.slots==24 and model.numeric==(0,1,2)
    assert not any(hasattr(model,name) for name in ('device','logger','scaler','_scaler','task_level'))
    snapshot={key:value.clone() for key,value in model.named_buffers()};x=inputs(meta,3,6);before=torch.get_rng_state().clone()
    actual=model(dict(X=x,y=torch.tensor(float('nan'))))
    separate=torch.cat([model(dict(X=x[i:i+1])) for i in range(3)])
    torch.testing.assert_close(actual,separate,rtol=1e-10,atol=1e-10)
    changed=x.detach().clone();changed[1:,:,0,:3]+=100
    torch.testing.assert_close(actual[:1],model(dict(X=changed))[:1],rtol=0,atol=0)
    assert torch.equal(before,torch.get_rng_state())
    for key,value in model.named_buffers(): torch.testing.assert_close(value,snapshot[key],rtol=0,atol=0)
    actual[0].sum().backward();assert torch.count_nonzero(x.grad[1:])==0
    numeric,times=model.prepare_inputs(x);assert numeric.shape[-1]==3 and len(times)==2
    clock=torch.arange(6,dtype=torch.float64)[None,:,None]/24
    with torch.no_grad(): x[...,meta.input_layout['calendar']['time_in_day']]=clock
    _,times=model.prepare_inputs(x)
    torch.testing.assert_close(times[0],model.T_i_D_emb[torch.arange(6)][None,:,None].expand(3,-1,4,-1),rtol=0,atol=0)


def test_calendar_in_middle_keeps_following_external_numeric_columns():
    torch.manual_seed(17);meta=metadata(nodes=2,numeric=2,outputs=1)
    meta=replace(meta,feature_dim=12)
    model=D2STGNN(meta,5,1,**SMALL).double().eval()
    x=inputs(meta,2,5);numeric,times=model.prepare_inputs(x)
    assert model.numeric==(0,1,10,11) and numeric.shape[-1]==4
    torch.testing.assert_close(numeric,x[...,[0,1,10,11]],rtol=0,atol=0)
    assert model.gap==1 and model.out_fc_2.out_features==1
    assert model(dict(X=x)).shape==(2,1,2,1)


def test_raw_history_nan_zero_cache_and_future_statistics_isolation(tmp_path):
    folder=d2stgnn_fixture(tmp_path/'raw');path=folder/'DecoupleTiny.dyna';frame=pd.read_csv(path);times=sorted(frame['time'].unique())
    frame.loc[(frame['entity_id']==71)&frame['time'].isin([times[35],times[36]]),'value_0']=float('nan')
    frame.loc[(frame['entity_id']==3)&(frame['time']==times[36]),'value_0']=0.;frame.to_csv(path,index=False)
    context=make_context(configuration(tmp_path,'data'))
    try:
        first=context.bundle;sample=first.val.dataset[0];scaler=deepcopy(vars(first.scaler))
        assert torch.isfinite(sample['X']).all() and torch.isnan(sample['y_raw']).any() and torch.any(sample['y_raw']==0)
        frame.loc[frame['time']>=times[36],['value_0','value_1']]+=1000;frame.to_csv(path,index=False)
        changed=make_context(configuration(tmp_path,'changed'))
        try:
            assert vars(changed.bundle.scaler)==scaler
            torch.testing.assert_close(sample['X'],changed.bundle.val.dataset[0]['X'],rtol=0,atol=0)
            assert not torch.equal(sample['y_raw'],changed.bundle.val.dataset[0]['y_raw'])
        finally: changed.tracker.close()
        warmed=make_context(configuration(tmp_path,'warm'))
        try: assert warmed.bundle.split_summary['cache']['status']=='hit'
        finally: warmed.tracker.close()
    finally: context.tracker.close()


@pytest.mark.parametrize('singleton',[False,True])
def test_cpu_amp_real_tail_odd_hidden_single_node_and_all_graphs_disabled(tmp_path,singleton):
    if singleton: atomic_fixture(tmp_path/'raw',name='DecoupleTiny',steps=60,nodes=1)
    else: d2stgnn_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,'amp');cfg['training'].update(epochs=1,amp=True,batch_size=8)
    if singleton:
        cfg['data'].update(input_window=1,graph=dict(name='none'))
        cfg['model']['params'].update(k_t=1,num_hidden=3,attention_heads=1,use_time_of_day=False,use_day_of_week=False,dy_graph=False,sta_graph=False)
    result=run_experiment('train',config=cfg);assert np.isfinite(list(result.metrics.values())).all()


def test_windows_spawn_and_changed_curriculum_resume_is_rejected(tmp_path):
    d2stgnn_fixture(tmp_path/'raw');cfg=configuration(tmp_path,'spawn');cfg['training']['num_workers']=1
    context=make_context(cfg)
    try: assert next(iter(context.bundle.train))['X'].shape==(5,6,4,11)
    finally: context.tracker.close()
    cfg['runtime']['exp_id']='trained';cfg['training'].update(num_workers=0,epochs=1)
    full=run_experiment('train',config=cfg)
    cfg['runtime']['exp_id']='changed';cfg['training']['strategy']['cl_decay_epochs']=2
    with pytest.raises(ValueError,match='data/training configuration'):
        run_experiment('train',config=cfg,resume=full.artifacts['last'])


@pytest.mark.parametrize('change',[
    dict(model=dict(params=dict(num_hidden=0))),dict(model=dict(params=dict(node_dim=0))),dict(model=dict(params=dict(k_s=0))),
    dict(model=dict(params=dict(k_t=0))),dict(model=dict(params=dict(num_layers=0))),dict(model=dict(params=dict(gap=0))),
    dict(model=dict(params=dict(dropout=1.))),dict(model=dict(params=dict(attention_heads=3))),dict(model=dict(params=dict(steps_per_day=-1))),
    dict(model=dict(params=dict(dy_graph='yes'))),dict(model=dict(params=dict(time_emd_dim=10))),dict(model=dict(params=dict(dynamic_channel=-1))),
    dict(data=dict(input_window=2)),dict(data=dict(features=dict(time_in_day=False))),
    dict(training=dict(strategy=dict(cl_decay_epochs=0))),dict(training=dict(strategy=dict(use_curriculum_learning='yes')))])
def test_invalid_d2stgnn_configuration_rejected_before_data(change):
    with pytest.raises((ValueError,TypeError)):
        load_config(config=merge_config(dict(model=dict(name='D2STGNN'),data=dict(dataset='not-read')),change))
