"""Actual calendar, dual-stream scaling and repaired dynamic temporal graphs."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.data.dmstgcn_bundle import prepare_dmstgcn
from libcity.model.traffic_speed_prediction.runtime_dmstgcn import DMSTGCN,GraphConv,dynamic_graph
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,periodic_fixture,dmstgcn_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(nodes=4,features=3,outputs=1,slots=24):
    return DatasetMetadata('tiny',nodes,features,outputs,input_layout=dict(kind='window',
        observed_feature_indices=list(range(features)),
        history_features=dict(kind='calendar_positions',steps_per_day=slots),
        dmstgcn_features=dict(primary_index=0,auxiliary_index=1)))


def positions(batch,steps,slots=24):
    value = torch.zeros(batch,steps,2,dtype=torch.long)
    value[...,1] = (torch.arange(steps)[None]+torch.arange(batch)[:,None]*3)%slots
    return value


def configuration(root,name):
    return dict(data=dict(dataset='Tiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=2,pad_training=False),
        model=dict(name='DMSTGCN',params=dict(num_blocks=2,residual_channels=4,dilation_channels=4,
            end_channels=16,embedding_dims=4,skip_channels=4,dropout=.2)),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2),evaluation=dict(batch_size=2),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


@pytest.mark.parametrize('residual,dilation,end,embedding,blocks,layers,kernel,steps,horizon,norm,training,dropout',[
    (32,32,512,40,4,2,2,12,12,'batch',False,.3),
    (4,6,16,4,2,2,2,9,3,'batch',True,.2),
    (32,32,512,40,4,2,2,12,12,'layer',False,.3),
    (3,5,12,3,1,3,3,4,2,'layer',True,.1)])
def test_repaired_original_full_network_forecast_all_effective_gradients_and_bn(
        monkeypatch,residual,dilation,end,embedding,blocks,layers,kernel,steps,horizon,norm,training,dropout):
    module = importlib.import_module('libcity.model.traffic_speed_prediction.DMSTGCN')
    torch.manual_seed(17)
    # Legacy Conv1d has a two-dimensional kernel and receives BCHT input.
    with monkeypatch.context() as scope:
        scope.setattr(module.nn,'Conv1d',nn.Conv2d)
        old = module.DMSTGCN(dict(output_window=horizon,num_blocks=blocks,num_layers=layers,
            kernel_size=kernel,residual_channels=residual,dilation_channels=dilation,end_channels=end,
            embedding_dims=embedding,normalization='batch',dropout=dropout),
            dict(num_nodes=4,feature_dim=3,output_dim=1,time_slots=24,scaler=NoneScaler()))
    current = DMSTGCN(metadata=metadata(),input_window=steps,output_window=horizon,num_blocks=blocks,
        num_layers=layers,kernel_size=kernel,residual_channels=residual,dilation_channels=dilation,
        end_channels=end,embedding_dims=embedding,normalization=norm,dropout=dropout)
    old.end_conv_1 = nn.Conv2d(8*sum(current.lengths),end,1)
    for index,length in enumerate(current.lengths):
        if norm=='layer':
            old.normal[index] = nn.LayerNorm((residual,4,length))
            old.normal_a[index] = nn.LayerNorm((residual,4,length))
        if residual!=dilation:
            old.gconv_a2p[index] = module.gcn(residual,residual,dropout,support_len=1,order=2)
    old,current = old.double().train(training),current.double().train(training)
    loaded = current.load_state_dict(old.state_dict(),strict=False)
    assert not loaded.missing_keys
    source = torch.randn(2,steps,4,3,dtype=torch.float64,requires_grad=True)
    reference = source.detach().clone().requires_grad_()
    calendar = positions(2,steps)
    torch.manual_seed(99)
    expected = old.predict(dict(X=reference,idx=calendar[:,0,1]))
    torch.manual_seed(99)
    actual = current(dict(X=source,history_features=calendar))
    torch.testing.assert_close(actual,expected,rtol=2e-8,atol=2e-10)
    actual.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(source.grad,reference.grad,rtol=4e-7,atol=2e-10)
    previous = dict(old.named_parameters())
    for key,value in current.named_parameters():
        assert value.grad is not None and previous[key].grad is not None
        torch.testing.assert_close(value.grad,previous[key].grad,rtol=4e-7,atol=3e-10)
    old_buffers = dict(old.named_buffers())
    for key,value in current.named_buffers():
        torch.testing.assert_close(value,old_buffers[key],rtol=2e-12,atol=2e-12)
    assert not any(name.startswith('residual_convs') for name,_ in current.named_parameters())
    assert len(current.gconv)==current.count-1 and len(current.normal_a)==max(0,current.count-2)


def test_dynamic_graph_tensor_contraction_and_all_embedding_gradients_match_direct_formula():
    time = torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    source = torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    target = torch.randn(4,3,dtype=torch.float64,requires_grad=True)
    core = torch.randn(3,3,3,dtype=torch.float64,requires_grad=True)
    actual = dynamic_graph(time,source,target,core)
    logits = torch.stack([torch.stack([torch.einsum('i,j,k,ijk->',t,a,b,core)
        for b in target]) for t in time for a in source]).reshape(2,4,4)
    expected = logits.relu().softmax(-1)
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    first = torch.autograd.grad(actual.square().mean(),(time,source,target,core))
    second = torch.autograd.grad(expected.square().mean(),(time,source,target,core))
    for value,reference in zip(first,second,strict=True):
        torch.testing.assert_close(value,reference,rtol=1e-12,atol=1e-13)


@pytest.mark.parametrize('order',(0,1,2,3))
def test_graph_convolution_uses_original_directed_powers_and_channel_order(order):
    layer = GraphConv(2,3,0.,order).double()
    source = torch.randn(2,2,4,3,dtype=torch.float64,requires_grad=True)
    graph = torch.randn(2,4,4,dtype=torch.float64,requires_grad=True)
    powers = [torch.eye(4,dtype=torch.float64).expand(2,-1,-1)]
    for _ in range(order):
        powers.append(graph@powers[-1])
    propagated = torch.cat([torch.einsum('bnm,bcmt->bcnt',matrix,source) for matrix in powers],1)
    expected = layer.mlp(propagated)
    actual = layer(source,graph)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    parameters = [source,layer.mlp.mlp.weight,layer.mlp.mlp.bias]+([graph] if order else [])
    first = torch.autograd.grad(actual.square().mean(),parameters)
    second = torch.autograd.grad(expected.square().mean(),parameters)
    for value,reference in zip(first,second,strict=True):
        torch.testing.assert_close(value,reference,rtol=1e-11,atol=1e-12)


def test_actual_calendar_train_only_auxiliary_scaling_cache_and_future_isolation(tmp_path):
    folder = periodic_fixture(tmp_path/'raw',name='Tiny',steps=60,nodes=4,features=3,outputs=1)
    path = folder/'Tiny.dyna'
    frame = pd.read_csv(path)
    frame['time'] = (pd.to_datetime(frame['time'],utc=True)+pd.Timedelta(hours=7,minutes=30)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.to_csv(path,index=False)
    (folder/'Tiny.rel').write_text('unused malformed relation',encoding='utf-8')
    cfg = configuration(tmp_path,'data')
    cfg['data']['features'] = dict(time_in_day=True)
    def prepare(config):
        resolved = load_config(config=config)
        return prepare_dmstgcn(resolved['data'],resolved['training'],resolved['evaluation'],17)
    first = prepare(cfg)
    raw = first.train.dataset.calendar.windows
    assert first.split_summary['cache']['status']=='miss'
    info = first.split_summary['dmstgcn_features']
    mean,std = raw.history[:36,:,1].mean(),raw.history[:36,:,1].std()
    assert info['extra_scalers']['1']['state']==dict(mean=mean,std=std,scale=std)
    sample = first.train.dataset[0]
    assert sample['history_features'].dtype==torch.long and sample['history_features'][0,1]==7
    indices = raw.target_starts[0]+raw.offsets
    np.testing.assert_allclose(sample['X'][...,1],(raw.history[indices,:,1]-mean)/std,rtol=2e-6,atol=2e-6)
    changed = configuration(tmp_path,'changed')
    changed['data'].update(input_window=2,output_window=1,pad_training=True,features=dict(time_in_day=True,auxiliary_index=2))
    second = prepare(changed)
    assert second.split_summary['cache']['status']=='hit'
    assert '2' in second.split_summary['dmstgcn_features']['extra_scalers']
    future = pd.to_datetime(frame['time'],utc=True)>=pd.Timestamp(info['scaler_fit_end_exclusive'],tz='UTC')
    frame.loc[future,['value_0','value_1','value_2']] += 999
    frame.to_csv(path,index=False)
    after = prepare(cfg)
    assert after.split_summary['dmstgcn_features']==info
    for key in ('X','history_features'):
        torch.testing.assert_close(after.val.dataset[0][key],first.val.dataset[0][key],rtol=0,atol=0)
    assert not torch.equal(after.val.dataset[0]['y_raw'],first.val.dataset[0]['y_raw'])
    # Known calendar channels never silently become the second observation.
    invalid = configuration(tmp_path,'calendar')
    invalid['data']['features'] = dict(time_in_day=True,auxiliary_index=3)
    with pytest.raises(ValueError,match='historical observation'):
        prepare(invalid)


@pytest.mark.parametrize('normalization',('batch','layer'))
def test_model_is_label_free_ignores_unselected_features_and_is_batch_independent(normalization):
    model = DMSTGCN(metadata=metadata(outputs=2),input_window=4,output_window=3,num_blocks=2,
        residual_channels=3,dilation_channels=5,end_channels=12,embedding_dims=3,skip_channels=3,
        normalization=normalization,dropout=0.).double().eval()
    source = torch.randn(2,4,4,3,dtype=torch.float64,requires_grad=True)
    calendar = positions(2,4)
    before = {key:value.clone() for key,value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    actual = model(dict(X=source,history_features=calendar,y=torch.tensor(float('nan')),
        future_features=torch.tensor(float('nan'))))
    assert actual.shape==(2,3,4,2)
    changed = source.detach().clone();changed[...,2] = float('nan')
    torch.testing.assert_close(actual,model(dict(X=changed,history_features=calendar)),rtol=0,atol=0)
    separate = torch.cat([model(dict(X=value[None],history_features=time[None])) for value,time in zip(source,calendar,strict=True)])
    torch.testing.assert_close(actual,separate,rtol=2e-12,atol=2e-12)
    actual[0].sum().backward()
    assert source.grad[0,...,:2].abs().sum()>0 and not source.grad[1].any() and not source.grad[...,2].any()
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert not any(hasattr(model,key) for key in ('device','_scaler','_logger','adj_mx'))


@pytest.mark.parametrize('blocks,layers,order,steps',[(1,1,2,1),(1,2,0,1),(2,2,3,19)])
def test_single_layer_zero_order_and_long_history_have_only_effective_parameters(blocks,layers,order,steps):
    model = DMSTGCN(metadata=metadata(nodes=1,outputs=2),input_window=steps,output_window=3,
        num_blocks=blocks,num_layers=layers,residual_channels=2,dilation_channels=3,end_channels=8,
        embedding_dims=2,order=order,skip_channels=2,dropout=0.).double()
    source = torch.randn(1,steps,1,3,dtype=torch.float64,requires_grad=True)
    value = model(dict(X=source,history_features=positions(1,steps)))
    assert value.shape==(1,3,1,2) and torch.isfinite(value).all()
    value.square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    if blocks*layers==1 or order==0:
        assert not any(name.startswith('nodevec') for name,_ in model.named_parameters())


def test_forecast_head_keeps_horizon_and_target_channels_in_the_declared_order():
    model = DMSTGCN(metadata=metadata(outputs=2),input_window=2,output_window=3,num_blocks=1,
        residual_channels=2,dilation_channels=3,end_channels=4,embedding_dims=2,skip_channels=2).eval()
    with torch.no_grad():
        model.end_conv_2.weight.zero_()
        model.end_conv_2.bias.copy_(torch.arange(6.))
    actual = model(dict(X=torch.randn(2,2,4,3),history_features=positions(2,2)))
    expected = torch.arange(6.).reshape(1,3,1,2).expand(2,3,4,2)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def equal(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        for a,b in zip(left,right,strict=True):
            equal(a,b)
    else:
        assert left==right


@pytest.mark.parametrize('normalization',('batch','layer'))
def test_full_dual_stream_dropout_bn_adam_metric_schedule_and_rng_resume_exactly(tmp_path,normalization):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=60,nodes=4,features=3,outputs=2)
    def config(name):
        result = configuration(tmp_path,name)
        result['data']['features'] = dict(auxiliary_index=2)
        result['model']['params']['normalization'] = normalization
        return result
    trained = run_experiment('train',config=config('full'))
    evaluated = run_experiment('evaluate',config=config('evaluate'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    partial = make_context(config('partial'))
    before = partial.model.nodevec_pk.detach().clone()
    from libcity.executor.train import train_loop
    train_loop(partial,until_epoch=1)
    assert not torch.equal(before,partial.model.nodevec_pk)
    partial.tracker.close()
    resumed = run_experiment('train',config=config('resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(2,4,2)


@pytest.mark.parametrize('singleton_time',(False,True))
def test_actual_cpu_amp_single_node_true_tail_batch(singleton_time,tmp_path):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=50,nodes=1,features=2,outputs=1)
    cfg = configuration(tmp_path,'amp')
    cfg['data']['output_window'] = 1
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    if singleton_time:
        cfg['data']['input_window'] = 1
        cfg['model']['params'].update(kernel_size=1,num_blocks=1)
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


def test_spawn_worker_can_collate_scaled_streams_and_calendar(tmp_path):
    dmstgcn_fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'spawn')
    cfg['training']['num_workers'] = 1
    context = make_context(cfg)
    try:
        batch = next(iter(context.bundle.train))
        assert batch['X'].shape==(5,4,4,2) and batch['history_features'].dtype==torch.long
    finally:
        context.tracker.close()


@pytest.mark.parametrize('features',[dict(primary_index=-1),dict(auxiliary_index=0),dict(auxiliary_index=True)])
def test_invalid_stream_configuration_is_rejected(features):
    with pytest.raises(ValueError):
        load_config(config=dict(model=dict(name='DMSTGCN'),data=dict(dataset='Tiny',features=features)))


@pytest.mark.parametrize('params',[dict(num_layers=0),dict(num_blocks=0),dict(kernel_size=0),dict(embedding_dims=0),
    dict(order=-1),dict(dropout=1),dict(normalization='invalid'),dict(skip_channels=0)])
def test_invalid_model_structure_is_rejected(params):
    with pytest.raises(ValueError,match='DMSTGCN'):
        DMSTGCN(metadata=metadata(),input_window=4,output_window=2,**params)


def test_invalid_metadata_inputs_and_single_observation_dataset_fail_clearly(tmp_path):
    with pytest.raises(ValueError,match='history_features'):
        DMSTGCN(metadata=DatasetMetadata('missing',4,3,1,input_layout=dict(kind='window')),input_window=4,output_window=2)
    model = DMSTGCN(metadata=metadata(),input_window=4,output_window=2,num_blocks=1,residual_channels=2,
        dilation_channels=2,end_channels=8,embedding_dims=2)
    source = torch.zeros(1,4,4,3)
    for time in (None,torch.zeros(1,4,2),torch.zeros(1,3,2,dtype=torch.long)):
        with pytest.raises(ValueError,match='history_features'):
            model(dict(X=source,history_features=time))
    calendar = positions(1,4);calendar[0,0,1] = 24
    with pytest.raises(ValueError,match='calendar slots'):
        model(dict(X=source,history_features=calendar))
    atomic_fixture(tmp_path/'raw')
    cfg = load_config(config=configuration(tmp_path,'one'))
    with pytest.raises(ValueError,match='two distinct historical'):
        prepare_dmstgcn(cfg['data'],cfg['training'],cfg['evaluation'],17)
