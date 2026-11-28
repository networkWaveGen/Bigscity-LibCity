"""ESG's evolving relations, temporal scales and train-only graph history."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config,merge_config
from libcity.data.bundle import DatasetMetadata
from libcity.data.esg_bundle import prepare_esg
from libcity.data.sources import atomic_sources
from libcity.executor.train import train_loop
from libcity.model.traffic_flow_prediction.runtime_esg import ESG,EvolvingGraphLearner,MixProp
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,grid_fixture
from runtime_helpers import make_context

pytestmark=pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(nodes=4,features=3,outputs=2,rows=24):
    values=np.random.default_rng(17).normal(size=(rows,nodes)).astype(np.float32)
    return DatasetMetadata('tiny',nodes,features,outputs,input_layout=dict(kind='window'),static_data=dict(graph_history=values))


SMALL=dict(n_blocks=1,n_layers=2,dropout=0.,residual_channels=4,end_channels=7,st_embedding_dim=5,
    kernel_set=[2,3],dilation_exp=1,conv_channels=4,gcn_depth=2,dy_embedding_dim=3,skip_channels=5,
    layer_norm_affline=False,propalpha=.05,dy_interval=[1,2])


@pytest.mark.parametrize('case',[
    (2,4,3,2,4,3,24,{}),
    (1,3,2,1,11,2,25,dict(n_blocks=2,n_layers=2,kernel_set=[1,3],dilation_exp=2,conv_channels=6,
        residual_channels=3,layer_norm_affline=True,dy_interval=[2,3])),
    (2,3,2,1,12,2,23,dict(n_layers=3,residual_channels=32,end_channels=128,st_embedding_dim=40,
        kernel_set=[2,6],conv_channels=32,gcn_depth=2,dy_embedding_dim=20,skip_channels=64,dy_interval=[1,1,1])),
    (2,2,2,2,1,4,12,dict(n_blocks=2,n_layers=2,kernel_set=[1],gcn_depth=0,dy_interval=[5,5]))],
    ids=['multitarget','multiblock-dilation-affine','default-widths','zero-hop'])
def test_original_multiscale_network_predictions_all_live_gradients_and_batchnorm(case):
    batch,nodes,features,outputs,history,horizon,rows,changes=case
    torch.set_num_threads(1);torch.manual_seed(17)
    params={**SMALL,**changes};meta=metadata(nodes,features,outputs,rows)
    current=ESG(meta,history,horizon,**params).double()
    legacy=importlib.import_module('libcity.model.traffic_flow_prediction.ESG').ESG
    prior=meta.static_data['graph_history'] if params['gcn_depth'] else np.zeros((24,nodes),dtype=np.float32)
    old=legacy(dict(params,input_window=history,output_window=horizon,device=torch.device('cpu')),
        dict(num_nodes=nodes,feature_dim=features,output_dim=outputs,static_feat=prior)).double()
    old.static_feat=old.static_feat.double()
    loaded=old.load_state_dict({key:value for key,value in current.state_dict().items() if key!='static_feat'},strict=False)
    assert not loaded.unexpected_keys
    assert all('residual_conv.' in key or params['gcn_depth']==0 and (key.startswith('stfea_encode.') or
        '.s_conv.linear_s2d.' in key or '.s_conv.scale_spc_EGL.' in key) for key in loaded.missing_keys)
    x=torch.randn(batch,history,nodes,features,dtype=torch.float64,requires_grad=True)
    other=x.detach().clone().requires_grad_()
    actual=current(dict(X=x));expected=old.predict(dict(X=other))
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    coefficients=torch.linspace(.3,1.2,actual.numel(),dtype=actual.dtype).reshape_as(actual)
    (actual*coefficients).sum().backward();(expected*coefficients).sum().backward()
    for name,parameter in current.named_parameters():
        assert parameter.grad is not None,name
        torch.testing.assert_close(parameter.grad,dict(old.named_parameters())[name].grad,rtol=1e-7,atol=1e-9)
    torch.testing.assert_close(x.grad,other.grad,rtol=1e-7,atol=1e-9)
    for key,value in current.state_dict().items():
        if key.startswith('stfea_encode.') and ('running_' in key or 'num_batches' in key):
            torch.testing.assert_close(value,old.state_dict()[key],rtol=1e-12,atol=1e-12)
    assert not any('residual_conv' in name for name,_ in current.named_parameters())
    if not params['gcn_depth']: assert current.stfea_encode is None and current.static_feat is None


def test_evolving_graph_matches_independent_gates_pair_scores_and_all_gradients():
    torch.manual_seed(17);model=EvolvingGraphLearner(5,4).double()
    source=torch.randn(2,3,5,dtype=torch.float64,requires_grad=True)
    states=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
    actual,updated=model(source,states)
    gates=F.linear(torch.cat((source,states),-1),model.rz_gate.weight,model.rz_gate.bias).sigmoid()
    reset,update=gates[...,:4],gates[...,4:]
    candidate=F.linear(torch.cat((source,reset*states),-1),model.h_candidate.weight,model.h_candidate.bias).tanh()
    expected_state=update*states+(1-update)*candidate
    matrices=[]
    for batch in range(2):
        rows=[]
        for receiver in range(3):
            values=[]
            for sender in range(3):
                pair=torch.cat((expected_state[batch,receiver].relu(),expected_state[batch,sender].relu()))
                def score(layer): return F.linear(F.linear(pair,layer.fc2.weight,layer.fc2.bias).relu(),layer.fc1.weight,layer.fc1.bias).squeeze()
                values.append(score(model.conv)*score(model.conv2).sigmoid())
            rows.append(torch.stack(values))
        matrices.append(torch.stack(rows))
    expected=torch.stack(matrices)
    torch.testing.assert_close(updated,expected_state,rtol=0,atol=0)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    first=torch.autograd.grad(actual.square().sum(),(source,states),retain_graph=True)
    second=torch.autograd.grad(expected.square().sum(),(source,states))
    for a,b in zip(first,second): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize('alpha',[0.,.25,1.])
def test_signed_graph_propagation_receiver_direction_matches_numpy(alpha):
    torch.manual_seed(17);model=MixProp(2,3,2,alpha).double()
    source=torch.randn(2,2,3,4,dtype=torch.float64)
    graph=torch.tensor([[[.3,-.4,.2],[.5,.1,-.2],[-.1,.7,.4]]],dtype=torch.float64).expand(2,-1,-1)
    actual=model(source,graph)
    values=source.numpy();states=[values]
    for _ in range(2):
        values=alpha*source.numpy()+(1-alpha)*np.stack([[graph[b].numpy()@values[b,c] for c in range(2)] for b in range(2)])
        states.append(values)
    concatenated=np.concatenate(states,axis=1)
    weights=model.mlp.mlp.weight.detach().numpy()[...,0,0];bias=model.mlp.mlp.bias.detach().numpy()
    expected=np.stack([[sum(weights[out,c]*concatenated[b,c] for c in range(6))+bias[out] for out in range(3)] for b in range(2)])
    np.testing.assert_allclose(actual.detach(),expected,rtol=1e-12,atol=1e-12)


def configuration(root,name='run',grid=True):
    return dict(model=dict(name='ESG',params={**SMALL,'dropout':.2}),
        data=dict(dataset='GridTiny' if grid else 'Tiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=2,pad_training=False),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
            scheduler=dict(name='plateau',interval='metric',monitor='val_loss',factor=.5,patience=0)),
        evaluation=dict(batch_size=4),runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def bundle(cfg): return prepare_esg(cfg['data'],cfg['training'],cfg['evaluation'],17)


@pytest.mark.parametrize('grid',[False,True])
def test_graph_prior_is_original_training_prefix_independent_of_windows_and_future(tmp_path,grid):
    folder=grid_fixture(tmp_path/'raw',steps=60) if grid else atomic_fixture(tmp_path/'raw',steps=60)
    (folder/(('GridTiny' if grid else 'Tiny')+'.rel')).write_text('not a usable graph',encoding='utf-8')
    cfg=configuration(tmp_path,grid=grid)
    cfg['data']['features']=dict(time_in_day=True,day_in_week=True,load_external=grid,known_future_columns=['holiday'] if grid else [])
    cfg=load_config(config=cfg);first=bundle(cfg);windows=first.train.dataset.windows
    raw=windows.arrays['history'] if grid else windows.history
    expected=first.scaler.transform(raw[:36,:,:first.metadata.output_dim]).transpose(0,2,1).reshape(-1,first.metadata.num_nodes).astype(np.float32)
    np.testing.assert_equal(first.metadata.static_data['graph_history'],expected)
    assert not first.metadata.static_data['graph_history'].flags.writeable
    cfg['data'].update(input_window=2,output_window=3,pad_training=True)
    changed_window=bundle(cfg)
    np.testing.assert_equal(first.metadata.static_data['graph_history'],changed_window.metadata.static_data['graph_history'])
    assert first.split_summary['cache']['path']==changed_window.split_summary['cache']['path']
    sample=first.val.dataset[0];assert sample['X'].shape[-1]==first.metadata.feature_dim
    model=ESG(first.metadata,4,2,**SMALL).eval();prediction=model({key:value.unsqueeze(0) for key,value in sample.items()})
    suffix='.grid' if grid else '.dyna';path=folder/(('GridTiny' if grid else 'Tiny')+suffix)
    frame=pd.read_csv(path);future=frame['time']>=sorted(frame['time'].unique())[36]
    columns=['value_0','value_1'] if grid else ['speed'];frame.loc[future,columns]+=1000;frame.to_csv(path,index=False)
    cfg['data'].update(input_window=4,output_window=2,pad_training=False);changed=bundle(cfg)
    np.testing.assert_equal(first.metadata.static_data['graph_history'],changed.metadata.static_data['graph_history'])
    torch.testing.assert_close(sample['X'],changed.val.dataset[0]['X'],rtol=0,atol=0)
    torch.testing.assert_close(prediction,model({key:value.unsqueeze(0) for key,value in changed.val.dataset[0].items()}),rtol=0,atol=0)
    assert not torch.equal(sample['y_raw'],changed.val.dataset[0]['y_raw'])
    if grid:
        assert any(row['name']=='GridTiny.grid' and row['size'] for row in atomic_sources(cfg['data']))
        assert not any(row['name']=='GridTiny.dyna' and row['size'] is None for row in atomic_sources(cfg['data']))


def test_graph_history_channel_limit_cache_hit_and_owned_buffer_dtype(tmp_path,monkeypatch):
    grid_fixture(tmp_path/'raw',steps=60);cfg=load_config(config=configuration(tmp_path));first=bundle(cfg)
    original=pd.read_csv
    def guarded(path,*args,**kwargs):
        if Path(path).suffix in ('.grid','.ext') and kwargs.get('nrows')!=0: pytest.fail('warm cache read complete atoms')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',guarded)
    cfg['data']['graph'].update(history_steps=20,history_channels=[1]);second=bundle(cfg)
    assert second.split_summary['cache']['status']=='hit' and second.metadata.static_data['graph_history'].shape==(20,6)
    raw=second.train.dataset.windows.arrays['history']
    np.testing.assert_equal(second.metadata.static_data['graph_history'],second.scaler.transform(raw[:20,:,1]).astype(np.float32))
    model=ESG(second.metadata,4,2,**SMALL).double()
    assert model.static_feat.dtype==torch.float64 and model.static_feat is dict(model.named_buffers())['static_feat']
    assert not any(hasattr(model,name) for name in ('device','scaler','_scaler','logger'))
    sample={key:value[None].double() for key,value in second.val.dataset[0].items()}
    model.eval();before=torch.get_rng_state();result=model(sample)
    torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
    sample['future_features']=torch.randn(1,2,100,dtype=torch.float64);sample['y']=torch.randn(1,2,6,2,dtype=torch.float64)
    torch.testing.assert_close(result,model(sample),rtol=0,atol=0)


def equal(left,right):
    if isinstance(left,torch.Tensor): torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True): equal(a,b)
    else: assert left==right


@pytest.mark.parametrize('scaler',['standard','minmax01'])
@pytest.mark.parametrize('grid',[False,True])
def test_real_dropout_batchnorm_raw_evaluation_and_all_state_exact_resume(tmp_path,scaler,grid):
    if grid: grid_fixture(tmp_path/'raw',steps=60)
    else: atomic_fixture(tmp_path/'raw',steps=60)
    def config(name):
        cfg=configuration(tmp_path,name,grid);cfg['data']['scaler']=dict(name=scaler);return cfg
    full=run_experiment('train',config=config('full'))
    evaluated=run_experiment('evaluate',config=config('eval'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as source:
        prediction,truth=source['prediction'],source['truth'];valid=np.isfinite(truth)&(truth!=0)
        assert prediction.shape==((11,2,6,2) if grid else (11,2,3,1))
        assert evaluated.metrics['loss']==pytest.approx(np.abs(prediction[valid]-truth[valid]).mean(),rel=2e-6,abs=1e-7)
    assert not (tmp_path/'runs/eval/checkpoints').exists() and not (tmp_path/'runs/eval/metrics.jsonl').exists()
    partial=make_context(config('partial'));before=partial.model.blocks[0][0].s_conv.scale_spc_EGL.rz_gate.weight.detach().clone()
    try:
        train_loop(partial,until_epoch=1)
        assert not torch.equal(before,partial.model.blocks[0][0].s_conv.scale_spc_EGL.rz_gate.weight)
    finally: partial.tracker.close()
    resumed=run_experiment('train',config=config('resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual=[torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'): equal(expected[key],actual[key])


@pytest.mark.parametrize('grid',[False,True])
def test_missing_history_is_imputed_while_original_zero_and_nan_targets_survive(tmp_path,grid):
    folder=grid_fixture(tmp_path/'raw',steps=60) if grid else atomic_fixture(tmp_path/'raw',steps=60)
    path=folder/('GridTiny.grid' if grid else 'Tiny.dyna');frame=pd.read_csv(path);times=sorted(frame['time'].unique())
    first=(frame['row_id']==0)&(frame['column_id']==0) if grid else frame['entity_id']==0
    second=(frame['row_id']==0)&(frame['column_id']==1) if grid else frame['entity_id']==1
    column='value_0' if grid else 'speed'
    frame.loc[frame['time'].isin([times[35],times[36]])&first,column]=float('nan')
    frame.loc[(frame['time']==times[36])&second,column]=0.;frame.to_csv(path,index=False)
    cfg=load_config(config=configuration(tmp_path,grid=grid));data=bundle(cfg);sample=data.val.dataset[0]
    assert torch.isfinite(sample['X']).all() and np.isfinite(data.metadata.static_data['graph_history']).all()
    assert torch.isnan(sample['y_raw']).any() and torch.any(sample['y_raw']==0)
    zero=sample['y_raw']==0;assert torch.all(sample['y'][zero]!=0)
    model=ESG(data.metadata,4,2,**SMALL).eval()
    assert torch.isfinite(model({name:value[None] for name,value in sample.items()})).all()


def test_physical_node_permutation_batch_independence_and_no_cross_sample_gradient(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(17)
    folder=grid_fixture(tmp_path/'raw',steps=60);cfg=load_config(config=configuration(tmp_path))
    first=bundle(cfg);model=ESG(first.metadata,4,2,**SMALL).eval()
    path=folder/'GridTiny.geo';geo=pd.read_csv(path);geo.iloc[[3,5,1,0,4,2]].to_csv(path,index=False)
    changed=bundle(cfg);other=ESG(changed.metadata,4,2,**SMALL).eval()
    loaded=other.load_state_dict({key:value for key,value in model.state_dict().items() if key!='static_feat'},strict=False)
    assert loaded.missing_keys==['static_feat'] and not loaded.unexpected_keys
    ids=list(first.metadata.info['node_ids']);order=[ids.index(value) for value in changed.metadata.info['node_ids']]
    a=torch.stack([first.val.dataset[i]['X'] for i in range(3)]).requires_grad_()
    b=torch.stack([changed.val.dataset[i]['X'] for i in range(3)])
    actual=model(dict(X=a))
    torch.testing.assert_close(actual[:,:,order],other(dict(X=b)),rtol=2e-6,atol=1e-7)
    torch.testing.assert_close(actual,torch.cat([model(dict(X=a[i:i+1])) for i in range(3)]),rtol=2e-6,atol=1e-7)
    gradient=torch.autograd.grad(actual[0].sum(),a)[0]
    assert torch.count_nonzero(gradient[1:])==0


@pytest.mark.parametrize('singleton',[False,True])
def test_true_cpu_amp_tail_and_single_node_minimum_encoder_history(tmp_path,singleton):
    grid_fixture(tmp_path/'raw',steps=60,rows=1 if singleton else 2,columns=1 if singleton else 3,features=1 if singleton else 3,outputs=1 if singleton else 2)
    cfg=configuration(tmp_path,'amp');cfg['training'].update(epochs=1,amp=True)
    if singleton: cfg['data']['graph']=dict(history_steps=19)
    result=run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


def test_short_graph_history_is_valid_for_zero_hop_but_rejected_when_graph_encoder_used(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=24,nodes=1)
    cfg=configuration(tmp_path,grid=False);cfg['data']['graph']=dict(history_steps=4)
    with pytest.raises(ValueError,match='19'): make_context(cfg)
    cfg['model']['params']['gcn_depth']=0;cfg['runtime']['exp_id']='zero-hop';cfg['training']['epochs']=1
    result=run_experiment('train',config=cfg)
    state=torch.load(result.artifacts['best'],weights_only=True)
    assert not any('stfea_encode.' in key or 'scale_spc_EGL.' in key or key=='static_feat' for key in state)


def test_graph_history_gap_rejects_or_can_select_contiguous_training_prefix(tmp_path):
    folder=grid_fixture(tmp_path/'raw',steps=60);path=folder/'GridTiny.grid';frame=pd.read_csv(path)
    times=sorted(frame['time'].unique());after=frame['time'].isin(times[25:])
    frame.loc[after,'time']=(pd.to_datetime(frame.loc[after,'time'],utc=True)+pd.Timedelta(hours=1)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.to_csv(path,index=False)
    cfg=load_config(config=configuration(tmp_path))
    with pytest.raises(ValueError,match='ESG graph history'): bundle(cfg)
    cfg['data']['graph']['history_steps']=20
    assert bundle(cfg).metadata.static_data['graph_history'].shape==(40,6)


def test_windows_spawn_worker_preserves_history_inputs(tmp_path):
    grid_fixture(tmp_path/'raw',steps=60);cfg=configuration(tmp_path,'spawn');cfg['training']['num_workers']=1
    context=make_context(cfg)
    try:
        batch=next(iter(context.bundle.train));assert batch['X'].shape==(5,4,6,3)
    finally: context.tracker.close()


@pytest.mark.parametrize('change',[
    dict(model=dict(params=dict(n_blocks=0))),dict(model=dict(params=dict(n_layers=0))),dict(model=dict(params=dict(gcn_depth=-1))),
    dict(model=dict(params=dict(kernel_set=[]))),dict(model=dict(params=dict(kernel_set=[6,2]))),dict(model=dict(params=dict(conv_channels=3))),
    dict(model=dict(params=dict(dy_interval=[1]))),dict(model=dict(params=dict(dy_interval=[1,0,1]))),dict(model=dict(params=dict(dropout=1.))),
    dict(model=dict(params=dict(propalpha=1.1))),dict(model=dict(params=dict(layer_norm_affline='yes'))),
    dict(data=dict(graph=dict(history_steps=-1))),dict(data=dict(graph=dict(history_channels=[0,0]))),
    dict(data=dict(graph=dict(name='weighted'))),dict(data=dict(graph=dict(self_loops=True)))])
def test_invalid_esg_configuration_rejected_before_sources(change):
    with pytest.raises((ValueError,TypeError)):
        load_config(config=merge_config(dict(model=dict(name='ESG'),data=dict(dataset='not-read')),change))
