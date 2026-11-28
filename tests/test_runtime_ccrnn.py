"""Coupled graph math, training-only SVD, explicit teacher forcing and RMSE."""
import importlib
import json
import random

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
import torch

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.data.ccrnn_bundle import initial_graph,prepare_ccrnn
from libcity.executor.ccrnn_train import build_loss,curriculum_step,train
from libcity.executor.curriculum import teacher_probability
from libcity.executor.evaluate import ForecastLoss,aggregate_loss
from libcity.executor.root_loss import RootMeanSquaredLoss
from libcity.executor.train import TrainState
from libcity.model.traffic_demand_prediction.runtime_ccrnn import CCRNN,GraphConv
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,periodic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(nodes=4,features=2,outputs=2):
    matrix = np.arange(nodes*nodes,dtype=np.float32).reshape(nodes,nodes)+1
    matrix /= matrix.sum(1,keepdims=True)
    return DatasetMetadata('tiny',nodes,features,outputs,adjacency=matrix,input_layout=dict(kind='window'))


def configuration(root,name):
    return dict(data=dict(dataset='Tiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=2,pad_training=False),
        model=dict(name='CCRNN',params=dict(hidden_size=4,n_dim=3,n_rnn_layers=2)),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,strategy=dict(cl_decay_steps=5)),
        evaluation=dict(batch_size=2),runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


@pytest.mark.parametrize('nodes,units,rank,layers,graph_layers,depth,features,outputs,steps,horizon,teacher',[
    (50,25,50,1,3,3,2,2,12,12,False),
    (4,4,3,2,3,2,3,2,4,3,True),
    (4,3,2,1,2,1,2,1,3,2,False),
    (4,3,2,2,1,3,2,2,2,3,True)])
def test_original_coupled_graph_forecast_teacher_and_all_effective_gradients(
        monkeypatch,nodes,units,rank,layers,graph_layers,depth,features,outputs,steps,horizon,teacher):
    original = importlib.import_module('libcity.model.traffic_demand_prediction.CCRNN')
    meta = metadata(nodes,features,outputs)
    monkeypatch.setattr(original,'graph_preprocess',lambda matrix,category:np.array(matrix,copy=True))
    torch.manual_seed(17)
    old = original.CCRNN(dict(device=torch.device('cpu'),hidden_size=units,n_dim=rank,k_hop=depth,
        n_rnn_layers=layers,n_gconv_layers=graph_layers,input_window=steps,output_window=horizon),
        dict(num_nodes=nodes,feature_dim=features,output_dim=outputs,adj_mx=meta.adjacency,scaler=NoneScaler())).double()
    current = CCRNN(metadata=meta,input_window=steps,output_window=horizon,hidden_size=units,n_dim=rank,
        k_hop=depth,n_rnn_layers=layers,n_gconv_layers=graph_layers).double()
    aliases = {f'graph_weights.{index}':f'w{index+1}' for index in range(graph_layers-1)}
    aliases.update({f'graph_biases.{index}':f'b{index+1}' for index in range(graph_layers-1)})
    previous = dict(old.named_parameters())
    with torch.no_grad():
        for key,value in current.named_parameters():
            value.copy_(previous[aliases.get(key,key)])
    source = torch.randn(2,steps,nodes,features,dtype=torch.float64,requires_grad=True)
    reference_source = source.detach().clone().requires_grad_()
    targets = torch.randn(2,horizon,nodes,outputs,dtype=torch.float64)
    choices = [index%2==0 for index in range(horizon)]
    draws = iter([0. if selected else 1. for selected in choices])
    monkeypatch.setattr(original.random,'random',lambda:next(draws))
    old.train(teacher)
    old._compute_sampling_threshold = lambda seen:.5
    expected = old(dict(X=reference_source,y=targets if teacher else None),batches_seen=3)
    actual = (current.forward_with_teacher(dict(X=source),targets,choices) if teacher else current(dict(X=source)))
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=3e-10)
    actual.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(source.grad,reference_source.grad,rtol=3e-7,atol=3e-10)
    for key,value in current.named_parameters():
        reference = previous[aliases.get(key,key)]
        assert value.grad is not None and reference.grad is not None
        torch.testing.assert_close(value.grad,reference.grad,rtol=3e-7,atol=5e-10)


@pytest.mark.parametrize('depth',(0,1,2,3))
def test_diffusion_convolution_matches_independent_matrix_polynomials_and_gradients(depth):
    layer = GraphConv(2,3,depth).double()
    source = torch.randn(2,4,2,dtype=torch.float64,requires_grad=True)
    support = torch.randn(4,4,dtype=torch.float64,requires_grad=True)*.1
    matrices = [torch.eye(4,dtype=torch.float64)]
    if depth:
        matrices.append(support)
    for _ in range(2,depth+1):
        matrices.append(2*support@matrices[-1]-matrices[-2])
    values = torch.stack([torch.einsum('nm,bmc->bnc',matrix,source) for matrix in matrices],-1).flatten(2)
    expected = values@layer.out.weight.T+layer.out.bias
    actual = layer(source,support)
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    parameters = [source,layer.out.weight,layer.out.bias]+([support] if depth else [])
    first = torch.autograd.grad(actual.square().mean(),parameters,retain_graph=True)
    second = torch.autograd.grad(expected.square().mean(),parameters)
    for value,reference in zip(first,second,strict=True):
        torch.testing.assert_close(value,reference,rtol=1e-12,atol=1e-13)


@pytest.mark.parametrize('normalization',('randomwalk','laplacian'))
def test_training_svd_gaussian_matches_original_distances_and_correct_normalization(normalization,monkeypatch):
    values = np.random.default_rng(17).normal(size=(21,5,2))
    observed = []
    original = np.linalg.svd
    def checked(matrix,**kwargs):
        assert kwargs['full_matrices'] is False
        result = original(matrix,**kwargs)
        observed.append((matrix.shape,result[0].shape,result[2].shape))
        return result
    monkeypatch.setattr(np.linalg,'svd',checked)
    actual,info = initial_graph(values,3,normalization)
    matrix = values.transpose(0,2,1).reshape(-1,5)
    _,singular,right = original(matrix,full_matrices=False)
    vectors = (np.diag(singular[:3])@right[:3]).T
    distance = cdist(vectors,vectors)
    graph = np.exp(-distance/distance.std()**2)-np.eye(5)
    degree = np.diag(graph.sum(1))
    if normalization=='randomwalk':
        expected = np.linalg.inv(degree)@graph
    else:
        inverse = np.diag(1/np.sqrt(graph.sum(1)))
        expected = np.eye(5)-inverse@graph@inverse
    np.testing.assert_allclose(actual,expected,rtol=2e-7,atol=2e-8)
    assert observed==[((42,5),(42,5),(5,5))]
    assert info==dict(requested_rank=3,actual_rank=3,constant_distance=False)


@pytest.mark.parametrize('nodes,normalization',[(1,'randomwalk'),(1,'laplacian'),(4,'randomwalk'),(4,'laplacian')])
def test_constant_observations_and_single_node_have_defined_initial_support(nodes,normalization):
    support,details = initial_graph(np.ones((30,nodes,2)),25,normalization)
    assert details['constant_distance'] and details['actual_rank']==nodes
    if nodes==1:
        expected = np.zeros((1,1)) if normalization=='randomwalk' else np.ones((1,1))
    else:
        transition = (np.ones((nodes,nodes))-np.eye(nodes))/(nodes-1)
        expected = transition if normalization=='randomwalk' else np.eye(nodes)-transition
    np.testing.assert_allclose(support,expected,rtol=1e-7,atol=1e-7)


@pytest.mark.parametrize('declared_columns',(True,False))
def test_data_uses_raw_training_interval_and_cache_ignores_unused_rel_windows_padding_and_rank(tmp_path,declared_columns):
    folder = periodic_fixture(tmp_path/'raw',name='Tiny',steps=60,nodes=4,features=3,outputs=2)
    if not declared_columns:
        path = folder/'config.json'
        info = json.loads(path.read_text(encoding='utf-8'))
        info.pop('data_col')
        path.write_text(json.dumps(info),encoding='utf-8')
    (folder/'Tiny.rel').write_text('unused malformed relation file',encoding='utf-8')
    cfg = configuration(tmp_path,'data')
    def prepare(config):
        resolved = load_config(config=config)
        return prepare_ccrnn(resolved['data'],resolved['training'],resolved['evaluation'],17)
    first = prepare(cfg)
    assert first.split_summary['cache']['status']=='miss'
    description = first.split_summary['graph_initialization']
    assert description['end_index_exclusive']==36 and description['observations_shape']==[36,4,3]
    expected,_ = initial_graph(first.train.dataset.history[:36,:,:3],25,'randomwalk')
    np.testing.assert_array_equal(first.metadata.adjacency,expected)
    alternative = configuration(tmp_path,'changed')
    alternative['data'].update(input_window=2,output_window=1,pad_training=True)
    alternative['training']['batch_size'] = 7
    second = prepare(alternative)
    assert second.split_summary['cache']['status']=='hit'
    np.testing.assert_array_equal(second.metadata.adjacency,first.metadata.adjacency)
    assert second.split_summary['graph_initialization']==description
    selected = configuration(tmp_path,'selected')
    selected['data']['graph'] = dict(svd_channels=[2],svd_rank=1,normalization='laplacian')
    third = prepare(selected)
    assert third.split_summary['cache']['status']=='hit'
    expected,_ = initial_graph(third.train.dataset.history[:36,:,2:3],1,'laplacian')
    np.testing.assert_array_equal(third.metadata.adjacency,expected)
    assert not first.metadata.adjacency.flags.writeable
    frame = pd.read_csv(folder/'Tiny.dyna')
    future = pd.to_datetime(frame['time'],utc=True)>=pd.Timestamp(description['last_timestamp'],tz='UTC')+pd.Timedelta(hours=1)
    frame.loc[future,['value_0','value_1','value_2']] += 999
    frame.to_csv(folder/'Tiny.dyna',index=False)
    after = prepare(cfg)
    np.testing.assert_array_equal(after.metadata.adjacency,first.metadata.adjacency)
    torch.testing.assert_close(after.val.dataset[0]['X'],first.val.dataset[0]['X'],rtol=0,atol=0)
    assert not torch.equal(after.val.dataset[0]['y_raw'],first.val.dataset[0]['y_raw'])


def test_calendar_graph_channels_are_rejected(tmp_path):
    atomic_fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'invalid')
    cfg['data']['graph'] = dict(svd_channels=[1])
    resolved = load_config(config=cfg)
    with pytest.raises(ValueError,match='historical observations'):
        prepare_ccrnn(resolved['data'],resolved['training'],resolved['evaluation'],17)


@pytest.mark.parametrize('nodes,depth,graph_layers',[(1,0,1),(1,3,3),(3,1,5),(4,3,1)])
def test_arbitrary_layers_zero_hop_small_graph_rank_and_dtype(nodes,depth,graph_layers):
    meta = metadata(nodes)
    model = CCRNN(metadata=meta,input_window=2,output_window=3,hidden_size=3,
        n_dim=50,k_hop=depth,n_rnn_layers=2,n_gconv_layers=graph_layers).double()
    source = torch.randn(1,2,nodes,2,dtype=torch.float64,requires_grad=True)
    prediction = model(dict(X=source))
    assert model.graph_dim==nodes and prediction.shape==(1,3,nodes,2)
    assert prediction.dtype==torch.float64 and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert torch.isfinite(source.grad).all()
    for value in model.parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all()
    if not depth:
        assert not hasattr(model,'nodevec1') and not model.graph_weights


def test_zero_support_keeps_zero_product_with_trainable_singular_direction():
    meta = DatasetMetadata('zero',2,1,1,adjacency=np.zeros((2,2)),input_layout=dict(kind='window'))
    model = CCRNN(metadata=meta,input_window=2,output_window=1,hidden_size=3,n_gconv_layers=1)
    assert not (model.nodevec1@model.nodevec2).any()
    output = model(dict(X=torch.ones(1,2,2,1)))
    output.sum().backward()
    assert model.nodevec2.grad.abs().sum()>0 and torch.isfinite(model.nodevec2.grad).all()


def test_forward_is_label_free_rng_free_sample_independent_and_has_no_graph_attributes():
    model = CCRNN(metadata=metadata(),input_window=3,output_window=2,hidden_size=4,n_dim=3).double()
    source = torch.randn(2,3,4,2,dtype=torch.float64,requires_grad=True)
    before = {key:value.clone() for key,value in model.state_dict().items()}
    torch_rng,python_rng = torch.get_rng_state().clone(),random.getstate()
    actual = model(dict(X=source,y=torch.full((1,),float('nan'))))
    model.eval()
    torch.testing.assert_close(actual,model(dict(X=source)),rtol=0,atol=0)
    separated = torch.cat([model(dict(X=value[None])) for value in source])
    torch.testing.assert_close(actual,separated,rtol=1e-12,atol=1e-12)
    actual[0].sum().backward()
    assert source.grad[0].abs().sum()>0 and not source.grad[1].any()
    assert torch.equal(torch_rng,torch.get_rng_state()) and python_rng==random.getstate()
    assert not any(hasattr(model,key) for key in ('graph0','graph1','graph2','adj_mx','device','_scaler','_logger'))
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_explicit_teacher_changes_later_steps_and_python_draws_follow_progress(tmp_path,monkeypatch):
    atomic_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'teacher'))
    try:
        batch = next(iter(context.bundle.train))
        source = context.loss_fn.inputs(batch)
        normal = context.model(source)
        teacher = context.model.forward_with_teacher(source,batch['y'],[True,True])
        torch.testing.assert_close(normal[:,0],teacher[:,0],rtol=0,atol=0)
        assert not torch.equal(normal[:,1],teacher[:,1])
        ignored = context.model.forward_with_teacher(source,torch.full_like(batch['y'],float('nan')),[False,False])
        torch.testing.assert_close(ignored,normal,rtol=0,atol=0)
        received = []
        original = context.model.forward_with_teacher
        def recorded(inputs,targets,choices):
            received.append(choices)
            return original(inputs,targets,choices)
        monkeypatch.setattr(context.model,'forward_with_teacher',recorded)
        strategy = context.config['training']['strategy']
        probability = teacher_probability(5,strategy['cl_decay_steps'])
        expected_rng = random.Random(31)
        expected = [expected_rng.random()<probability for _ in range(2)]
        random.seed(31)
        state = TrainState(batches_seen=5)
        result = curriculum_step(context.loss_fn,strategy)(context.model,batch,state)
        assert received==[expected] and random.getstate()==expected_rng.getstate()
        assert state.task_state['curriculum_batches_seen']==5 and torch.isfinite(result.loss)
        random_state = random.getstate()
        curriculum_step(context.loss_fn,{**strategy,'use_curriculum_learning':False})(context.model,batch,state)
        assert random.getstate()==random_state
        assert teacher_probability(10**9,300)==0
    finally:
        context.tracker.close()


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


@pytest.mark.parametrize('normalization',('randomwalk','laplacian'))
def test_real_svd_root_loss_train_evaluate_python_teacher_adam_metric_rng_exact_resume(tmp_path,normalization):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=60,nodes=4,features=2,outputs=2)
    def config(name):
        result = configuration(tmp_path,name)
        result['data']['graph'] = dict(normalization=normalization)
        return result
    full = run_experiment('train',config=config('full'))
    evaluated = run_experiment('evaluate',config=config('evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    partial = make_context(config('partial'))
    initial = partial.model.graph_weights[0].detach().clone()
    train(partial,until_epoch=1)
    assert not torch.equal(initial,partial.model.graph_weights[0])
    assert isinstance(partial.loss_fn,RootMeanSquaredLoss)
    partial.tracker.close()
    resumed = run_experiment('train',config=config('resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(2,4,2)
        residual = archive['prediction']-archive['truth']
        finite = np.isfinite(archive['truth'])
        assert evaluated.metrics['loss']==pytest.approx(float(np.sqrt(np.mean(residual[finite].astype(np.float64)**2))),rel=2e-7)


@pytest.mark.parametrize('nodes,depth',[(1,0),(1,3),(3,3)])
def test_real_cpu_amp_and_true_single_sample_tail(tmp_path,nodes,depth):
    atomic_fixture(tmp_path/'raw',steps=50,nodes=nodes)
    cfg = configuration(tmp_path,'amp')
    cfg['data']['output_window'] = 1
    cfg['model']['params'].update(n_dim=50,n_rnn_layers=1,k_hop=depth)
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


@pytest.mark.parametrize('change',[dict(svd_rank=0),dict(svd_channels=[]),dict(svd_channels=[0,0]),
    dict(normalization='invalid'),dict(name='binary'),dict(self_loops=True)])
def test_invalid_graph_policy_is_rejected(change):
    with pytest.raises(ValueError):
        load_config(config=dict(model=dict(name='CCRNN'),data=dict(dataset='Tiny',graph=change)))


@pytest.mark.parametrize('change',[dict(cl_decay_steps=0),dict(use_curriculum_learning=1),dict(unknown=1)])
def test_invalid_teacher_policy_is_rejected(change):
    with pytest.raises(ValueError):
        load_config(config=dict(model=dict(name='CCRNN'),data=dict(dataset='Tiny'),training=dict(strategy=change)))


@pytest.mark.parametrize('params',[dict(hidden_size=0),dict(n_dim=0),dict(k_hop=-1),dict(n_rnn_layers=0),dict(n_gconv_layers=0)])
def test_invalid_structure_is_rejected(params):
    with pytest.raises(ValueError,match='CCRNN'):
        CCRNN(metadata=metadata(),input_window=3,output_window=2,**params)


@pytest.mark.parametrize('threshold',(-1,float('nan'),True))
def test_plateau_threshold_is_validated_at_entry(threshold):
    with pytest.raises(ValueError):
        load_config(config=dict(model=dict(name='CCRNN'),data=dict(dataset='Tiny'),training=dict(scheduler=dict(threshold=threshold))))


def test_loss_factory_keeps_explicit_alternatives_and_rejects_shape_errors():
    bundle = type('Bundle',(),dict(scaler=NoneScaler(),metadata=metadata()))()
    assert isinstance(build_loss(bundle,dict(name='masked_rmse')),RootMeanSquaredLoss)
    assert type(build_loss(bundle,dict(name='mse'))) is ForecastLoss
    model = CCRNN(metadata=metadata(),input_window=3,output_window=2,hidden_size=3,n_dim=2)
    with pytest.raises(ValueError,match='X must'):
        model(dict(X=torch.zeros(1,2,4,2)))
    with pytest.raises(ValueError,match='teacher targets'):
        model.forward_with_teacher(dict(X=torch.zeros(1,3,4,2)),torch.zeros(1,1,4,2),[True])
