"""STSGCN graph math, trainable masks and the public forecast lifecycle."""
import importlib
import itertools
import re

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.executor.evaluate import ForecastLoss
from libcity.executor.train import train_loop
from libcity.model.traffic_flow_prediction.runtime_stsgcn import STSGCN,local_adjacency
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,periodic_fixture
from libcity.utils.normalization import NoneScaler,StandardScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def native_key(key):
    return re.sub(r'^(stsgcl_layers\.\d+)\.layer\.',r'\1.',key.replace('.ouput_layer.','.output_layer.'))


def metadata(nodes=3,features=2,outputs=2):
    graph = np.zeros((nodes,nodes),dtype=np.float32)
    if nodes>1:
        graph[0,1],graph[1,0] = .25,1.5
        graph[-1,0] = .6
    return DatasetMetadata('tiny',nodes,features,outputs,adjacency=graph,input_layout=dict(kind='window'))


def configuration(root,name,module_type='individual'):
    return dict(data=dict(dataset='Tiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=6,output_window=2,pad_training=False),
        model=dict(name='STSGCN',params=dict(module_type=module_type,filters=[[4,4],[5,5]],
            first_layer_embedding_size=3,output_hidden_size=8,use_mask=True)),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,clip_grad_norm=2.,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        evaluation=dict(batch_size=5),runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


@pytest.mark.parametrize('module_type,activation,options',list(itertools.product(
    ('individual','sharing'),('GLU','relu'),((4,True,True),(0,False,True),(0,True,False),(3,False,False)))))
def test_original_two_block_predictions_and_all_gradients_with_fresh_mask_graph(module_type,activation,options):
    embedding,temporal,spatial = options
    torch.manual_seed(19)
    meta = metadata()
    params = dict(input_window=6,output_window=3,filters=[[4,4],[5,5]],module_type=module_type,
        act_type=activation,first_layer_embedding_size=embedding,temporal_emb=temporal,spatial_emb=spatial,use_mask=True)
    old_type = importlib.import_module('libcity.model.traffic_flow_prediction.STSGCN').STSGCN
    old = old_type(dict(**params,device=torch.device('cpu')),
        dict(num_nodes=3,feature_dim=2,output_dim=2,adj_mx=meta.adjacency.copy(),scaler=NoneScaler())).double()
    with torch.no_grad():
        old.mask.mul_(.7)
    current = STSGCN(metadata=meta,**params).double()
    mapped = {native_key(key):value for key,value in old.state_dict().items()}
    keys = current.state_dict().keys()
    loaded = current.load_state_dict({key:value for key,value in mapped.items() if key in keys},strict=False)
    assert loaded.missing_keys==['adjacency'] and not loaded.unexpected_keys
    # The old initializer retained this graph forever; the reference recomputes it explicitly.
    old.adj = torch.from_numpy(local_adjacency(meta.adjacency)).double()*old.mask
    source = torch.randn(2,6,3,2,dtype=torch.float64,requires_grad=True)
    reference_source = source.detach().clone().requires_grad_()
    expected = old(dict(X=reference_source))
    actual = current(dict(X=source))
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(source.grad,reference_source.grad,rtol=1e-10,atol=1e-12)
    reference = {native_key(key):value for key,value in old.named_parameters()}
    for key,value in current.named_parameters():
        assert value.grad is not None
        torch.testing.assert_close(value.grad,reference[key].grad,rtol=1e-10,atol=2e-12)


def test_default_four_blocks_preserve_original_forecast_and_parameter_gradients():
    torch.manual_seed(23)
    meta = metadata(features=1,outputs=1)
    current = STSGCN(metadata=meta,input_window=12,output_window=2)
    old_type = importlib.import_module('libcity.model.traffic_flow_prediction.STSGCN').STSGCN
    old = old_type(dict(input_window=12,output_window=2,device=torch.device('cpu')),
        dict(num_nodes=3,feature_dim=1,output_dim=1,adj_mx=meta.adjacency.copy(),scaler=NoneScaler()))
    current.load_state_dict({native_key(key):value
        for key,value in old.state_dict().items()},strict=False)
    source = torch.randn(1,12,3,1)
    actual,expected = current(dict(X=source)),old(dict(X=source))
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    reference = {native_key(key):value for key,value in old.named_parameters()}
    for key,value in current.named_parameters():
        torch.testing.assert_close(value.grad,reference[key].grad,rtol=5e-4,atol=2e-6)


def test_directed_three_frame_graph_is_owned_and_follows_dtype():
    meta = metadata()
    original = meta.adjacency.copy()
    model = STSGCN(metadata=meta,input_window=3,output_window=2,filters=[[3]],use_mask=True).double()
    expected = np.zeros((9,9),dtype=np.float32)
    for block in range(3):
        expected[3*block:3*block+3,3*block:3*block+3] = original
    for block in range(2):
        for node in range(3):
            expected[block*3+node,(block+1)*3+node] = 1
            expected[(block+1)*3+node,block*3+node] = 1
    np.fill_diagonal(expected,1)
    np.testing.assert_array_equal(model.adjacency.numpy(),expected)
    assert 'adjacency' in dict(model.named_buffers()) and model.adjacency.dtype==torch.float64
    np.testing.assert_array_equal(meta.adjacency,original)
    assert all(not any(hasattr(part,key) for key in ('device','_device','_scaler','_logger')) for part in model.modules())


@pytest.mark.parametrize('symmetric',(False,True))
def test_inverse_distance_graph_preserves_missing_zero_and_directed_edges_with_cache(tmp_path,symmetric):
    folder = atomic_fixture(tmp_path/'raw',nodes=4)
    distances = pd.DataFrame(dict(rel_id=range(4),type='geo',origin_id=[0,1,2,3],
        destination_id=[1,0,0,1],distance=[2.,4.,0.,8.]))
    distances.to_csv(folder/'Tiny.rel',index=False)
    cfg = configuration(tmp_path,'graph')
    cfg['data']['graph'] = dict(name='inverse_distance',symmetric=symmetric,self_loops=False)
    expected = np.zeros((4,4),dtype=np.float32)
    expected[0,1],expected[1,0],expected[2,0],expected[3,1] = .5,.25,1.,.125
    if symmetric:
        expected = np.maximum(expected,expected.T)
    for status in ('miss','hit'):
        context = make_context(cfg)
        try:
            assert context.bundle.split_summary['cache']['status']==status
            np.testing.assert_array_equal(context.bundle.metadata.adjacency,expected)
        finally:
            context.tracker.close()
    cfg['data']['graph']['name'] = 'weighted'
    context = make_context(cfg)
    try:
        assert context.bundle.split_summary['cache']['status']=='miss'
        assert context.bundle.metadata.adjacency[3,1]==8.
    finally:
        context.tracker.close()


def test_unweighted_inverse_graph_fallback_does_not_create_missing_edges(tmp_path):
    folder = atomic_fixture(tmp_path/'raw',nodes=3)
    path = folder/'Tiny.rel'
    pd.read_csv(path).drop(columns='distance').to_csv(path,index=False)
    cfg = configuration(tmp_path,'binary-fallback')
    cfg['data']['graph'] = dict(name='inverse_distance',symmetric=False,self_loops=False)
    context = make_context(cfg)
    try:
        expected = np.roll(np.eye(3,dtype=np.float32),1,axis=1)
        np.testing.assert_array_equal(context.bundle.metadata.adjacency,expected)
    finally:
        context.tracker.close()


def test_negative_inverse_distances_fail_before_model_construction(tmp_path):
    folder = atomic_fixture(tmp_path/'raw')
    path = folder/'Tiny.rel'
    edges = pd.read_csv(path)
    edges.loc[0,'distance'] = -1.
    edges.to_csv(path,index=False)
    with pytest.raises(ValueError,match='nonnegative distances'):
        make_context(configuration(tmp_path,'negative-distance'))


@pytest.mark.parametrize('module_type',('individual','sharing'))
def test_repeated_mask_updates_are_fresh_without_retained_graphs(module_type):
    torch.manual_seed(11)
    model = STSGCN(metadata=metadata(),input_window=5,output_window=2,filters=[[3,3],[4,4]],
        first_layer_embedding_size=3,module_type=module_type,use_mask=True)
    initial_mask = model.mask.detach().clone()
    base = model.adjacency.clone()
    source = torch.randn(3,5,3,2)
    original = source.clone()
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(dict(X=source)).square().mean().backward()
        assert model.mask.grad.abs().sum()>0
        assert torch.count_nonzero(model.mask.grad[base==0])==0
        optimizer.step()
    assert not torch.equal(initial_mask,model.mask)
    torch.testing.assert_close(base,model.adjacency,rtol=0,atol=0)
    torch.testing.assert_close(source,original,rtol=0,atol=0)


@pytest.mark.parametrize('module_type',('individual','sharing'))
def test_label_free_batch_independence_float64_and_input_state_ownership(module_type):
    torch.manual_seed(3)
    model = STSGCN(metadata=metadata(),input_window=5,output_window=3,filters=[[3,3],[4,4]],
        first_layer_embedding_size=0,module_type=module_type,use_mask=True).double()
    source = torch.randn(2,5,3,2,dtype=torch.float64,requires_grad=True)
    original = source.detach().clone()
    state = {key:value.clone() for key,value in model.state_dict().items()}
    prediction = model(dict(X=source))
    torch.testing.assert_close(prediction[:1],model(dict(X=source[:1])),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(prediction,model(dict(X=source,y=torch.full_like(prediction,float('nan')),
        future_features=source.new_full((2,3,3,4),float('nan')))),rtol=0,atol=0)
    prediction[0].sum().backward()
    torch.testing.assert_close(source.grad[1],torch.zeros_like(source.grad[1]),rtol=0,atol=0)
    torch.testing.assert_close(source,original,rtol=0,atol=0)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,state[key],rtol=0,atol=0)


@pytest.mark.parametrize('delta,masked',list(itertools.product((.25,1.,2.5),(False,True))))
def test_raw_scale_huber_objective_count_and_double_precision_gradients(delta,masked):
    scaler = StandardScaler(10.,3.)
    raw = torch.tensor([0.,4.,8.,12.,float('nan')],dtype=torch.float64).reshape(1,1,5,1)
    normalized = scaler.transform(torch.tensor([-.2,4.1,9.5,18.,0.],dtype=torch.float64)).reshape_as(raw).requires_grad_()
    target = scaler.transform(raw)
    loss_fn = ForecastLoss(dict(name='masked_huber' if masked else 'huber',delta=delta,mask_value=0.),scaler,1)
    actual = loss_fn(normalized,target,raw)
    valid = torch.isfinite(raw)&((raw!=0) if masked else torch.ones_like(raw,dtype=torch.bool))
    reference = normalized.detach().clone().requires_grad_()
    expected = torch.nn.functional.huber_loss(scaler.inverse_transform(reference)[valid],raw[valid],delta=delta)
    assert actual.weight==int(valid.sum()) and actual.loss.dtype==torch.float64
    torch.testing.assert_close(actual.loss,expected,rtol=0,atol=0)
    actual.loss.backward()
    expected.backward()
    torch.testing.assert_close(normalized.grad,reference.grad,rtol=0,atol=0)
    empty = loss_fn(normalized,torch.full_like(target,float('nan')))
    assert empty.weight==0 and empty.loss.item()==0
    with pytest.raises(ValueError,match='non-finite'):
        loss_fn(torch.full_like(normalized,float('inf')),target,raw)


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


@pytest.mark.parametrize('module_type',('individual','sharing'))
def test_full_train_evaluate_and_exact_mask_adam_scheduler_rng_resume(tmp_path,module_type):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=48,features=2,outputs=2)
    trained = run_experiment('train',config=configuration(tmp_path,'full',module_type))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate',module_type),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    partial = make_context(configuration(tmp_path,'partial',module_type))
    initial = partial.model.mask.detach().clone()
    train_loop(partial,until_epoch=1)
    assert not torch.equal(partial.model.mask,initial)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume',module_type),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(2,4,2)


@pytest.mark.parametrize('module_type',('individual','sharing'))
def test_single_node_empty_spatial_graph_and_tail_cpu_amp_training(tmp_path,module_type):
    atomic_fixture(tmp_path/'raw',nodes=1)
    cfg = configuration(tmp_path,'amp',module_type)
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()
    model = STSGCN(metadata=metadata(nodes=1),input_window=3,output_window=2,filters=[[3,3]],use_mask=True).double()
    prediction = model(dict(X=torch.zeros(1,3,1,2,dtype=torch.float64)))
    assert prediction.shape==(1,2,1,2) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())


@pytest.mark.parametrize('delta',(0.,-1.,True,float('nan'),float('inf'),'1'))
def test_invalid_huber_threshold_is_rejected_before_data_loading(tmp_path,delta):
    config = configuration(tmp_path,'invalid')
    config['training']['loss'] = dict(name='huber',delta=delta)
    with pytest.raises(ValueError,match='training.loss.delta'):
        load_config(config=config)


def test_loss_selector_discards_huber_threshold_when_changing_objective(tmp_path):
    config = configuration(tmp_path,'loss')
    config['training']['loss'] = dict(name='mae')
    assert load_config(config=config)['training']['loss']==dict(name='mae')
    config['training']['loss']['delta'] = 2.
    with pytest.raises(ValueError,match='Unknown configuration keys'):
        load_config(config=config)


@pytest.mark.parametrize('params',[
    dict(filters=[]),dict(filters=[[3,4]]),dict(filters=[[True]]),dict(filters=[[0]]),
    dict(filters=[3]),dict(first_layer_embedding_size=-1),dict(output_hidden_size=0),
    dict(module_type='invalid'),dict(act_type='invalid'),dict(use_mask=1)])
def test_invalid_model_structure_is_rejected(params):
    with pytest.raises(ValueError,match='STSGCN'):
        STSGCN(metadata=metadata(),input_window=12,output_window=2,**params)


def test_invalid_window_graph_layout_and_forward_shape_are_rejected():
    with pytest.raises(ValueError,match='input_window'):
        STSGCN(metadata=metadata(),input_window=8,output_window=2)
    with pytest.raises(ValueError,match='output_window'):
        STSGCN(metadata=metadata(),input_window=12,output_window=0)
    meta = DatasetMetadata('bad',1,1,1,adjacency=np.array([[float('nan')]]),input_layout=dict(kind='window'))
    with pytest.raises(ValueError,match='adjacency'):
        STSGCN(metadata=meta,input_window=12,output_window=2)
    meta = DatasetMetadata('bad',1,1,1,adjacency=np.zeros((1,1)),input_layout=dict(kind='cpt'))
    with pytest.raises(ValueError,match='continuous window'):
        STSGCN(metadata=meta,input_window=12,output_window=2)
    model = STSGCN(metadata=metadata(),input_window=3,output_window=2,filters=[[3]])
    with pytest.raises(ValueError,match='X must'):
        model(dict(X=torch.zeros(1,4,3,2)))
