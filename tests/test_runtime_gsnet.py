"""Native GSNet preserves its two spatial modules and fixes forecast semantics."""
from dataclasses import replace
import importlib

import numpy as np
import pytest
import torch

from libcity.config.runtime import load_config,merge_config
from libcity.data.bundle import DatasetMetadata
from libcity.model.traffic_accident_prediction.runtime_gsnet import GSNet
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import risk_fixture
from libcity.executor.train import train_loop
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(rows=2,columns=3,features=3,external=2,future=4,steps=3,poi=True):
    cells = rows*columns
    graph_nodes = min(3,cells)
    generator = np.random.default_rng(17)
    mapping = np.zeros((cells,graph_nodes),dtype=np.float32)
    mapping[np.arange(graph_nodes)*cells//graph_nodes,np.arange(graph_nodes)] = 1.
    static = dict(grid_node_map=mapping,road_adj=generator.uniform(.1,1,(graph_nodes,graph_nodes)),
        risk_adj=generator.uniform(.1,1,(graph_nodes,graph_nodes)),risk_mask=np.ones(cells))
    if poi: static['poi_adj']=generator.uniform(.1,1,(graph_nodes,graph_nodes))
    return DatasetMetadata('tiny',cells,features,1,static_data=static,input_layout=dict(kind='risk_cpt',
        spatial_identity='row_major_grid_cells',grid_shape=[rows,columns],length=steps,
        historical_external_dim=external,future_external_dim=future,graph_nodes=graph_nodes,
        graph_input_indices=list(range(min(2,features)))))


def inputs(meta,batch,horizon,dtype=torch.float64):
    layout=meta.input_layout
    return dict(X=torch.rand(batch,layout['length'],meta.num_nodes,meta.feature_dim,dtype=dtype,requires_grad=True),
        history_features=torch.rand(batch,layout['length'],layout['historical_external_dim'],dtype=dtype,requires_grad=True),
        future_features=torch.rand(batch,horizon,layout['future_external_dim'],dtype=dtype,requires_grad=True))


@pytest.mark.parametrize('case',[
    (2,2,3,3,2,4,3,2,8,2,[4,6],True),
    (1,3,2,2,0,3,2,3,5,1,[7],True),
    (3,1,3,1,1,1,4,1,3,3,[2,4,5],False),
    (2,2,2,2,1,2,3,2,256,5,[64,64],True)],ids=['rectangular','batch-one','two-graphs','default-widths'])
def test_original_two_module_predictions_and_all_parameter_input_gradients(case):
    batch,rows,columns,features,external,future,steps,horizon,hidden,layers,filters,poi=case
    torch.set_num_threads(1);torch.manual_seed(17)
    meta=metadata(rows,columns,features,external,future,steps,poi)
    current=GSNet(meta,99,horizon,layers,hidden,filters).double()
    old=importlib.import_module('libcity.model.traffic_accident_prediction.GSNet')
    reference=old._GSNet(features+external,layers,steps,horizon,hidden,future,min(2,features),filters,rows,columns).double()
    outcome=reference.load_state_dict(current.gsnet.state_dict(),strict=poi)
    assert not outcome.unexpected_keys
    assert all(key.startswith('st_sem_module.poi_gcn.') for key in outcome.missing_keys)
    first=inputs(meta,batch,horizon)
    second={name:value.detach().clone().requires_grad_() for name,value in first.items()}
    observed=second['X'];values=observed
    if external: values=torch.cat((values,second['history_features'][:,:,None,:].expand(-1,-1,rows*columns,-1)),dim=-1)
    grid=values.permute(0,1,3,2).reshape(batch,steps,features+external,rows,columns)
    graph=(observed[...,:min(2,features)].permute(0,1,3,2)@current.grid_node_map)
    expected=reference(grid,second['future_features'][:,0],graph,current.road_adj,current.risk_adj,current.poi_adj,
        current.grid_node_map).reshape(batch,horizon,rows*columns,1)
    actual=current(first)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)
    coefficients=torch.linspace(.3,1.3,actual.numel(),dtype=actual.dtype).reshape_as(actual)
    (actual*coefficients).sum().backward();(expected*coefficients).sum().backward()
    old_parameters=dict(reference.named_parameters())
    for name,parameter in current.gsnet.named_parameters():
        assert parameter.grad is not None,name
        torch.testing.assert_close(parameter.grad,old_parameters[name].grad,rtol=1e-8,atol=1e-10)
    for name,value in first.items():
        if not external and name=='history_features': assert value.grad is None
        else: torch.testing.assert_close(value.grad,second[name].grad,rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize('future',[0,2])
def test_singleton_grid_time_batch_and_all_forecast_heads_keep_dimensions_and_gradients(future):
    torch.set_num_threads(1);torch.manual_seed(8)
    meta=metadata(1,1,1,0,future,1,False)
    model=GSNet(meta,1,3,1,4,[2]).double()
    batch=inputs(meta,1,3)
    prediction=model(batch)
    assert prediction.shape==(1,3,1,1)
    prediction.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.all(model.gsnet.output_layer.bias.grad==1)
    if not future:
        assert model.gsnet.st_geo_module.grid_att_fc2 is None
        assert model.gsnet.st_sem_module.graph_att_fc2 is None
        assert torch.equal(model(dict(X=batch['X'])),prediction)
    assert not any(name.startswith('gsnet.st_sem_module.poi_gcn.') for name,_ in model.named_parameters())
    for name in ('grid_node_map','road_adj','risk_adj'):
        assert name in dict(model.named_buffers()) and getattr(model,name).dtype==torch.float64
    assert not any(hasattr(model,name) for name in ('_scaler','scaler','logger','device'))


def test_forecast_context_is_first_known_future_time_without_targets_or_other_samples():
    torch.set_num_threads(1);torch.manual_seed(17)
    meta=metadata();model=GSNet(meta,12,3,2,8,[4,6]).double()
    batch=inputs(meta,3,3)
    captured=[]
    handle=model.gsnet.st_geo_module.grid_att_fc2.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    try: prediction=model(batch)
    finally: handle.remove()
    torch.testing.assert_close(captured[0],batch['future_features'][:,0],rtol=0,atol=0)
    changed={**batch,'future_features':batch['future_features'].detach().clone(),
        'y':torch.randn(3,3,6,1,dtype=torch.float64),'y_raw':torch.randn(3,3,6,1,dtype=torch.float64)}
    changed['future_features'][:,1:]+=1000
    torch.testing.assert_close(prediction,model(changed),rtol=0,atol=0)
    separate=torch.cat([model({name:value[index:index+1] for name,value in batch.items()}) for index in range(3)])
    torch.testing.assert_close(prediction,separate,rtol=1e-12,atol=1e-12)
    gradient=torch.autograd.grad(prediction[0].sum(),batch['X'])[0]
    assert torch.count_nonzero(gradient[1:])==0
    changed={name:value.detach().clone() for name,value in batch.items()};changed['future_features'][:,0]+=1
    assert not torch.equal(prediction,model(changed))


def test_semantic_graph_node_permutation_preserves_physical_grid_predictions():
    torch.set_num_threads(1);torch.manual_seed(17)
    meta=metadata();first=GSNet(meta,3,2,1,5,[3,4]).double()
    order=np.array([2,0,1]);static={name:np.array(value,copy=True) for name,value in meta.static_data.items()}
    for name in ('road_adj','risk_adj','poi_adj'): static[name]=static[name][order][:,order]
    static['grid_node_map']=static['grid_node_map'][:,order]
    second=GSNet(replace(meta,static_data=static),3,2,1,5,[3,4]).double()
    second.gsnet.load_state_dict(first.gsnet.state_dict())
    batch=inputs(meta,2,2)
    torch.testing.assert_close(first(batch),second(batch),rtol=1e-12,atol=1e-12)
    # Buffers own their storage; later caller changes cannot replace model priors.
    static['road_adj'][:]=999
    assert second.road_adj.max()<2


def configuration(root,name):
    return dict(model=dict(name='GSNet',params=dict(num_of_gru_layers=2,gru_hidden_size=8,gcn_nums_filters=[4,6])),
        data=dict(dataset='RiskTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),input_window=3,output_window=2,
            pad_training=False,sampling=dict(name='risk_cpt',closeness=2,period=1,trend=0,interval_period=1,interval_trend=1),
            features=dict(known_future_columns=['holiday'])),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
            scheduler=dict(name='plateau',interval='metric',monitor='val_loss',factor=.5,patience=0)),
        evaluation=dict(batch_size=4,metrics=['MAE','RMSE','Precision','Recall','F1-Score','MAP','PCC'],topk=2),
        runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def equal(left,right):
    if isinstance(left,torch.Tensor): torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True): equal(a,b)
    else: assert left==right


@pytest.mark.parametrize('scaler',['minmax01','standard'])
def test_native_risk_training_all_horizons_region_metrics_independent_evaluation_and_exact_resume(tmp_path,scaler):
    risk_fixture(tmp_path/'raw')
    def config(name):
        cfg=configuration(tmp_path,name);cfg['data']['scaler']=dict(name=scaler);return cfg
    full=run_experiment('train',config=config('full'))
    evaluated=run_experiment('evaluate',config=config('eval'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        prediction,truth,regions=archive['prediction'],archive['truth'],archive['regions']
        assert prediction.shape==(19,2,6,1) and regions.tolist()==[True,False,True,False,False,True]
        region=np.array([1.,0.,2.,0.,0.,.5]);weights=np.array([.05,.2,.25,.5])[np.searchsorted([0.,20.,28.],truth,side='left')]
        expected=((prediction-truth)**2*(region/region.mean())[None,None,:,None]**2*weights).mean()
        assert evaluated.metrics['loss']==pytest.approx(expected,rel=3e-6)
        for step in range(2):
            assert evaluated.metrics[f'MAE@{step+1}']==pytest.approx(np.abs(prediction[:,step,regions]-truth[:,step,regions]).mean(),rel=2e-6)
    weights=torch.load(full.artifacts['best'],weights_only=True)
    assert 'grid_node_map' in weights and 'poi_adj' in weights
    assert not (tmp_path/'runs/eval/checkpoints').exists() and not (tmp_path/'runs/eval/metrics.jsonl').exists()
    partial=make_context(config('partial'))
    before=partial.model.gsnet.output_layer.weight.detach().clone()
    try:
        batch=next(iter(partial.bundle.train));prediction=partial.model(partial.loss_fn.inputs(batch));prediction.retain_grad()
        partial.loss_fn(prediction,batch['y'],batch['y_raw']).loss.backward()
        assert all(prediction.grad[:,step].abs().sum()>0 for step in range(2))
        partial.model.zero_grad(set_to_none=True)
        # Reset loader RNG after this inspection so the full/resumed comparisons
        # start from the identical seeded shuffled order.
        partial.bundle.train.generator.manual_seed(17)
        train_loop(partial,until_epoch=1)
        assert not torch.equal(before,partial.model.gsnet.output_layer.weight)
    finally: partial.tracker.close()
    resumed=run_experiment('train',config=config('resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual=[torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'): equal(expected[key],actual[key])


@pytest.mark.parametrize('poi',[False,True])
def test_actual_native_cpu_amp_and_real_tail_batch_with_two_or_three_graphs(tmp_path,poi):
    risk_fixture(tmp_path/'raw',poi=poi)
    cfg=configuration(tmp_path,'amp');cfg['training'].update(epochs=1,amp=True)
    result=run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()
    state=torch.load(result.artifacts['last'],weights_only=True)
    assert state['train_state']['optimizer_steps']==4
    assert ('poi_adj' in state['model'])==poi


def test_spawn_worker_transports_native_risk_data(tmp_path):
    risk_fixture(tmp_path/'raw');cfg=configuration(tmp_path,'spawn');cfg['training']['num_workers']=1
    context=make_context(cfg)
    try:
        batch=next(iter(context.bundle.train))
        assert batch['X'].shape==(5,3,6,3) and batch['future_features'].shape==(5,2,32)
    finally: context.tracker.close()


@pytest.mark.parametrize('change',[
    dict(model=dict(params=dict(num_of_gru_layers=0))),dict(model=dict(params=dict(gru_hidden_size=0))),
    dict(model=dict(params=dict(gcn_nums_filters=[]))),dict(model=dict(params=dict(gcn_nums_filters=64))),
    dict(model=dict(params=dict(gcn_nums_filters=[4,False]))),dict(data=dict(features=dict(graph_columns=['risk','risk']))),
    dict(data=dict(features=dict(external_scaler=dict(name='broken')))),dict(data=dict(graph=dict(road_column=4))),
    dict(data=dict(graph=dict(use_poi='yes'))),dict(training=dict(loss=dict(name='mse'))),
    dict(training=dict(loss=dict(mask_value=0))),dict(training=dict(loss=dict(thresholds=[2,1]))),
    dict(training=dict(loss=dict(thresholds=[1],weights=[1,2,3]))),dict(training=dict(loss=dict(weights=[-1]))),
    dict(evaluation=dict(topk=0)),dict(evaluation=dict(metrics=['masked_Recall'])),dict(evaluation=dict(mask_value=0)),
    dict(data=dict(sampling=dict(name='window'))),dict(data=dict(graph=dict(name='none')))])
def test_invalid_gsnet_configuration_fails_before_reading_data(change):
    with pytest.raises((ValueError,TypeError)):
        load_config(config=merge_config(dict(model=dict(name='GSNet'),data=dict(dataset='not-read')),change))


def test_named_risk_protocol_preserves_original_recent_and_weekly_counts_and_raw_zero_risks():
    cfg=load_config(config=dict(model=dict(name='GSNet'),data=dict(dataset='not-read')))
    assert cfg['data']['sampling']==dict(name='risk_cpt',closeness=3,period=4,trend=0,interval_period=7,interval_trend=1)
    assert cfg['data']['output_window']==1 and cfg['evaluation']['mask_value'] is None
    assert cfg['training']['loss']==dict(name='risk_mse',mask_value=None,thresholds=None,weights=None)
    assert cfg['model']['params']==dict(num_of_gru_layers=5,gru_hidden_size=256,gcn_nums_filters=[64,64])
