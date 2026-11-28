"""ResLSTM corrections: independent samples, explicit periods and SAME pooling."""
import copy
import importlib
import itertools
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_flow_prediction.runtime_reslstm import ResLSTM,Unit,normalized_adjacency
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import periodic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,experiment):
    return dict(data=dict(dataset='PeriodicTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),pad_training=False,
            features=dict(time_in_day=True,day_in_week=True)),
        model=dict(name='ResLSTM',params=dict(base_channels=4,recurrent_hidden=8,recurrent_layers=2)),
        training=dict(epochs=3,batch_size=16,optimizer=dict(lr=.001),
            scheduler=dict(name='multistep',interval='epoch',milestones=[1,2],gamma=.8)),
        evaluation=dict(batch_size=17),runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


def corrected_legacy(old,source,history,flows,external_indices):
    size,_,nodes,_ = source.shape
    packed = [source[...,flow].reshape(size,3,history,nodes).permute(0,1,3,2) for flow in flows]
    graph = torch.tensor(old.adj_mx,dtype=source.dtype)@source[:,-history:,:,flows[0]].transpose(1,2)
    prediction = old.conv_block1(packed[0])+old.conv_block2(packed[1])+old.conv_block3(graph[:,None])
    if external_indices:
        external = source[:,-history:,0,list(external_indices)].transpose(1,2)
        prediction = prediction+old.fourth_pro(external)
    hidden,_ = old.lstm3(prediction[:,:,None])
    return old.fc_last(old.att(hidden).flatten(1)).reshape(size,1,nodes,1)


@pytest.mark.parametrize('external,flows',list(itertools.product((False,True),((0,1),(1,0)))))
def test_corrected_legacy_period_axes_same_pooling_batch_axes_and_all_gradients(tmp_path,external,flows):
    periodic_fixture(tmp_path/'raw',steps=800,outputs=1)
    config = configuration(tmp_path,'reference')
    config['data']['features'] = dict(time_in_day=external,day_in_week=external)
    config['model']['params'] = dict(base_channels=32,recurrent_hidden=128,recurrent_layers=2,flow_indices=list(flows))
    context = make_context(config)
    try:
        model = context.model.eval()
        metadata = context.bundle.metadata
        old_type = importlib.import_module('libcity.model.traffic_flow_prediction.ResLSTM').ResLSTM
        old = old_type(dict(time_lag=6,device=torch.device('cpu')),dict(num_nodes=4,output_dim=1,
            ext_dim=8 if external else 0,adj_mx=metadata.adjacency.copy(),scaler=NoneScaler())).eval()
        for part in old.modules():
            if isinstance(part,nn.LSTM):
                part.batch_first = True
            if isinstance(part,nn.MaxPool2d):
                part.padding = 0
                part.ceil_mode = True
        incoming = {key:value for key,value in old.state_dict().items() if key in model.state_dict()}
        loaded = model.load_state_dict(incoming,strict=False)
        assert loaded.missing_keys==['adj_mx'] and not loaded.unexpected_keys
        assert not any('unit1.conv1' in key for key in model.state_dict())
        batch = {key:value[:2] for key,value in next(iter(context.bundle.train)).items()}
        expected = corrected_legacy(old,batch['X'],5,flows,model.external_indices)
        actual = model(model_inputs(batch,1))
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        reference = dict(old.named_parameters())
        for key,value in model.named_parameters():
            assert value.grad is not None
            torch.testing.assert_close(value.grad,reference[key].grad,rtol=3e-4,atol=3e-6)
        assert all(not any(hasattr(part,key) for key in ('device','_device','_logger','_scaler')) for part in model.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('nodes,history',[(1,1),(1,2),(2,1),(2,2),(3,5),(3,6),(4,5),(4,6)])
def test_same_pooling_matches_explicit_negative_infinity_padding_and_gradients(nodes,history):
    torch.manual_seed(31)
    model = Unit(2,4,pool=True).double().eval()
    reference = copy.deepcopy(model)
    source = -torch.rand(1,2,nodes,history,dtype=torch.float64)
    padded = F.pad(source,(0,history%2,0,nodes%2),value=-torch.inf)
    pooled = F.max_pool2d(padded,2,2)
    residual = reference.conv1(source)
    expected = residual+reference.conv3(F.relu(reference.bn2(reference.conv2(F.relu(reference.bn1(pooled))))))
    actual = model(source)
    assert actual.shape==(1,4,(nodes+1)//2,(history+1)//2)
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for key,value in model.named_parameters():
        torch.testing.assert_close(value.grad,dict(reference.named_parameters())[key].grad,rtol=1e-12,atol=1e-12)


def equal(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True):
            equal(a,b)
    else:
        assert left==right


def test_full_train_evaluate_bn_adam_schedule_rng_exact_resume(tmp_path):
    periodic_fixture(tmp_path/'raw',steps=800)
    trained = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluated'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    info = json.loads(trained.artifacts['runtime_info'].read_text())
    assert info['benchmark_protocol']=='reslstm_windows'
    assert info['split']['history_span_points']==365 and info['split']['input_window']==15
    partial = make_context(configuration(tmp_path,'partial'))
    initial = {key:value.detach().clone() for key,value in partial.model.named_parameters()}
    train_loop(partial,until_epoch=1)
    changed = {key.split('.')[0] for key,value in partial.model.named_parameters() if not torch.equal(value,initial[key])}
    assert {'conv_block1','conv_block2','conv_block3','fc1','lstm1','lstm2','fc2','lstm3','att','fc_last'} <= changed
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape==(160,1,4,2)
    disabled = configuration(tmp_path,'evaluated')
    disabled['evaluation']['save_predictions'] = False
    result = run_experiment('evaluate',config=disabled,checkpoint=trained.artifacts['best'])
    assert 'predictions' not in result.artifacts and not (result.artifacts['metrics_json'].parent/'predictions.npz').exists()


def test_node_and_batch_axes_are_independent_in_evaluation_and_gradients(tmp_path):
    periodic_fixture(tmp_path/'raw',steps=800)
    context = make_context(configuration(tmp_path,'independent'))
    try:
        model = context.model.eval().double()
        source = next(iter(context.bundle.test))['X'][:2].double()
        single = model(dict(X=source[:1]))
        mixed = source.clone()
        mixed[1] = mixed[1]*20+100
        actual = model(dict(X=mixed))
        torch.testing.assert_close(single,actual[:1],rtol=1e-11,atol=1e-11)
        mixed.requires_grad_()
        model(dict(X=mixed))[0].sum().backward()
        torch.testing.assert_close(mixed.grad[1],torch.zeros_like(mixed.grad[1]),rtol=0,atol=0)
        assert torch.count_nonzero(mixed.grad[0])>0
    finally:
        context.tracker.close()


@pytest.mark.parametrize('nodes,history,external',[(1,1,False),(1,2,True),(3,2,False),(4,5,True)])
def test_derived_shapes_multistep_multitarget_batch_one_float64_and_stateless_buffers(tmp_path,nodes,history,external):
    periodic_fixture(tmp_path/'raw',steps=80,nodes=nodes,features=3,outputs=3)
    cfg = configuration(tmp_path,'dimensions')
    cfg['data'].update(input_window=history,output_window=3,sampling=dict(period_steps=6,trend_steps=12),
                       features=dict(time_in_day=external,day_in_week=external))
    cfg['model']['params'].update(base_channels=2,recurrent_hidden=4,recurrent_layers=1,flow_indices=[2,0])
    context = make_context(cfg)
    try:
        model = context.model.double().eval()
        batch = {key:value[:1] for key,value in next(iter(context.bundle.test)).items()}
        source = batch['X'].double()
        original = source.clone()
        state = {key:value.clone() for key,value in model.state_dict().items()}
        rng = torch.get_rng_state()
        predicted = model(dict(X=source))
        assert predicted.shape==(1,3,nodes,3) and predicted.dtype==torch.float64
        torch.testing.assert_close(predicted,model(dict(X=source,y=torch.full_like(predicted,float('nan')))),rtol=0,atol=0)
        equal(model.state_dict(),state)
        torch.testing.assert_close(source,original,rtol=0,atol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        predicted.sum().backward()
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())
        assert hasattr(model,'fc1')==external
        packed = model.packed_flows(source)
        for segment in range(3):
            for time in range(history):
                torch.testing.assert_close(packed[:,segment,:,time,:],source[:,segment*history+time],rtol=0,atol=0)
    finally:
        context.tracker.close()


def test_external_schema_uses_only_historical_global_features_and_not_extra_raw_observations(tmp_path):
    folder = periodic_fixture(tmp_path/'raw',steps=800,features=3)
    times = pd.read_csv(folder/'PeriodicTiny.dyna')['time'].drop_duplicates().tolist()
    pd.DataFrame(dict(ext_id=range(len(times)),time=times,temperature=np.arange(len(times))*.01,
        holiday=np.arange(len(times))%2)).to_csv(folder/'PeriodicTiny.ext',index=False)
    cfg = configuration(tmp_path,'external')
    cfg['data']['features'].update(load_external=True,known_future_columns=['holiday'])
    context = make_context(cfg)
    try:
        layout = context.bundle.metadata.input_layout
        assert layout['reslstm_observation_dim']==3
        assert list(layout['reslstm_external_indices'])==list(range(3,13))
        model = context.model.eval().double()
        batch = {key:value[:1].double() for key,value in next(iter(context.bundle.test)).items()}
        initial = model(model_inputs(batch,2))
        # A third raw node-level observation is not silently mistaken for a
        # shared external covariate; flows are explicitly selected by the model.
        changed = {key:value.clone() for key,value in batch.items()}
        changed['X'][...,2] += 100
        changed['future_features'].fill_(1e6)
        changed['y'].fill_(float('nan'))
        torch.testing.assert_close(initial,model(model_inputs(changed,2)),rtol=0,atol=0)
        source = changed['X'].detach().requires_grad_()
        gradient = torch.autograd.grad(model(dict(X=source)).sum(),source)[0]
        assert torch.count_nonzero(gradient[:,-5:,0,list(model.external_indices)])>0
        torch.testing.assert_close(gradient[...,2],torch.zeros_like(gradient[...,2]),rtol=0,atol=0)
    finally:
        context.tracker.close()


def test_training_normalization_and_input_predictions_ignore_changed_future_targets(tmp_path):
    folder = periodic_fixture(tmp_path/'raw',steps=800)
    first = make_context(configuration(tmp_path,'causal'))
    try:
        before = first.bundle.test.dataset[0]
        start = int(first.bundle.test.dataset.target_starts[0])
        stamp = (pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(hours=start)).strftime('%Y-%m-%dT%H:%M:%SZ')
        frame = pd.read_csv(folder/'PeriodicTiny.dyna')
        frame.loc[frame.time==stamp,['value_0','value_1']] = 1e6
        frame.to_csv(folder/'PeriodicTiny.dyna',index=False)
        second = make_context(configuration(tmp_path,'modified'))
        try:
            after = second.bundle.test.dataset[0]
            assert second.bundle.split_summary['cache']['status']=='miss'
            assert vars(first.bundle.scaler)==vars(second.bundle.scaler)
            torch.testing.assert_close(before['X'],after['X'],rtol=0,atol=0)
            assert not torch.equal(before['y_raw'],after['y_raw'])
            first.model.eval()
            torch.testing.assert_close(first.model(dict(X=before['X'][None])),first.model(dict(X=after['X'][None])),rtol=0,atol=0)
        finally:
            second.tracker.close()
    finally:
        first.tracker.close()


def test_raw_column_inference_and_missing_flow_rejection(tmp_path):
    folder = periodic_fixture(tmp_path/'raw',steps=800)
    metadata_path = folder/'config.json'
    raw = json.loads(metadata_path.read_text())
    del raw['data_col']
    metadata_path.write_text(json.dumps(raw))
    context = make_context(configuration(tmp_path,'inferred'))
    assert context.bundle.metadata.input_layout['reslstm_observation_dim']==2
    context.tracker.close()
    periodic_fixture(tmp_path/'single'/'raw',steps=800,features=1,outputs=1)
    with pytest.raises(ValueError,match='two distinct raw flow'):
        make_context(configuration(tmp_path/'single','rejected'))


def test_normalized_graph_matches_independent_column_degree_formula_and_owns_data():
    adjacency = np.array([[0,1,2],[0,0,1],[1,0,0]],dtype=np.float64)
    original = adjacency.copy()
    expected = adjacency+np.eye(3)
    inverse = np.diag(1/np.sqrt(expected.sum(0)))
    np.testing.assert_allclose(normalized_adjacency(adjacency),inverse@expected@inverse,rtol=1e-15,atol=1e-15)
    np.testing.assert_array_equal(adjacency,original)
    np.testing.assert_array_equal(normalized_adjacency(np.zeros((1,1))),np.ones((1,1)))


def test_singleton_true_tail_bn_cpu_amp_training(tmp_path):
    periodic_fixture(tmp_path/'raw',steps=40,nodes=1)
    cfg = configuration(tmp_path,'amp')
    cfg['data'].update(input_window=1,sampling=dict(period_steps=6,trend_steps=12))
    cfg['training'].update(epochs=1,batch_size=5,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()
    state = torch.load(result.artifacts['last'],weights_only=True)
    assert state['train_state']['optimizer_steps']==3


@pytest.mark.parametrize('parameters',[dict(base_channels=0),dict(recurrent_layers=0),dict(flow_indices=[0,0]),
                                    dict(flow_indices=[0,2]),dict(flow_indices=[[],0])])
def test_invalid_dimensions_and_flow_indices_are_rejected(parameters):
    metadata = DatasetMetadata('tiny',1,2,1,adjacency=np.zeros((1,1)),input_layout=dict(kind='seasonal_windows',length=15,
        reslstm_observation_dim=2,reslstm_external_indices=[]))
    with pytest.raises(ValueError,match='ResLSTM'):
        ResLSTM(metadata=metadata,input_window=5,output_window=1,**parameters)


def test_reslstm_rejects_a_contiguous_window_protocol():
    with pytest.raises(ValueError,match='seasonal_windows'):
        load_config(config=dict(data=dict(dataset='unused',sampling=dict(name='window')),model=dict(name='ResLSTM')))


@pytest.mark.parametrize('name',['ResLSTM','CONVGCN'])
def test_service_windows_allow_between_window_closures_but_keep_each_segment_and_targets_complete(tmp_path,name):
    folder = periodic_fixture(tmp_path/'raw',steps=100)
    frame = pd.read_csv(folder/'PeriodicTiny.dyna')
    original = sorted(frame['time'].unique())
    service = [(pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(days=index//18,hours=6+index%18)).strftime('%Y-%m-%dT%H:%M:%SZ')
               for index in range(100)]
    frame['time'] = frame['time'].map(dict(zip(original,service)))
    frame.to_csv(folder/'PeriodicTiny.dyna',index=False)
    config = configuration(tmp_path,'service')
    if name=='CONVGCN':
        config['model'] = dict(name=name,params=dict(conv_depth=2,conv_height=2,hidden_size=4))
    config['data'].update(input_window=3,output_window=2,sampling=dict(period_steps=18,trend_steps=36))
    context = make_context(config)
    try:
        assert context.bundle.metadata.input_layout['gap_policy']=='seasonal_segments_v1'
        assert context.bundle.split_summary['sampling_gap_policy']=='seasonal_segments_and_recent_target_contiguous'
        for loader in (context.bundle.train,context.bundle.val,context.bundle.test):
            dataset = loader.dataset
            clock = dataset.history  # Keep raw arrays owned by the shared cache.
            assert len(clock)==100
            times = np.asarray(service,dtype='datetime64[s]')
            assert any(times[start]-times[start+dataset.offsets.min()]>np.timedelta64(39,'h') for start in dataset.target_starts)
            for start in dataset.target_starts:
                for segment in dataset.offsets.reshape(3,3):
                    assert np.all(np.diff(times[start+segment])==np.timedelta64(1,'h'))
                assert np.all(np.diff(times[start-3:start+2])==np.timedelta64(1,'h'))
        batch = next(iter(context.bundle.test))
        context.model.eval()
        prediction = context.model(model_inputs(batch,2))
        assert prediction.shape==batch['y'].shape and torch.isfinite(prediction).all()
    finally:
        context.tracker.close()
