"""STPGCN registration, Gaussian factorization, actual calendar and lifecycle."""
import importlib
import re

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse.csgraph import shortest_path
from torch import nn

from libcity.data.bundle import DatasetMetadata
from libcity.data.stpgcn_bundle import calendar_positions
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_flow_prediction.runtime_stpgcn import STPGCN,RelationInference,bounded_hops
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,periodic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(nodes=4,features=2,outputs=2):
    graph = np.zeros((nodes,nodes),dtype=np.float32)
    if nodes>1:
        graph[np.arange(nodes-1),np.arange(1,nodes)] = 1
    return DatasetMetadata('tiny',nodes,features,outputs,adjacency=graph,info=dict(time_intervals=1800),
        input_layout=dict(kind='window',history_features=dict(kind='calendar_positions',
            columns=['weekday','time_slot'],steps_per_day=48)))


def configuration(root,name):
    return dict(data=dict(dataset='Tiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=2),
        model=dict(name='STPGCN',params=dict(C=4,L=2,d=2,alpha=1,beta=2)),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,clip_grad_norm=2.,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        evaluation=dict(batch_size=5),runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


def native_key(key):
    for before,after in {
        'SAPE.sape':'spatial_embedding','TAPE.dow_emb':'week_embedding','TAPE.tod_emb':'day_embedding',
        'SRPE.srpe':'spatial_relative_embedding','TRPE.trpe':'temporal_relative_embedding',
        'net.input_layer.fc':'input_projection','net.fs.':'initial_selection.',
        'net.glu':'fusion','net.output_layer.fc':'output_projection'}.items():
        key = key.replace(before,after)
    for before,after in {'ri_lst':'relation','gc_lst':'convolution','fs_lst':'selection'}.items():
        key = re.sub(r'^net\.'+before+r'\.(\d+)\.',r'layers.\1.'+after+'.',key)
    return key.replace('.gc_lst.','.components.')


def corrected_reference(meta,current,params,input_window,output_window):
    source = importlib.import_module('libcity.model.traffic_flow_prediction.STPGCN')
    reference = source.STPGCN(dict(device=torch.device('cpu'),input_window=input_window,
        output_window=output_window,**params),dict(output_dim=meta.output_dim,feature_dim=meta.feature_dim,
        num_nodes=meta.num_nodes,points_per_hour=2,scaler=NoneScaler(),
        spatial_distance=np.zeros((meta.num_nodes,meta.num_nodes,1),dtype=np.int64),
        range_mask=np.tile(current.range_mask.numpy(),(1,params['beta']+1))))
    # Correct only the documented registration/hop/sentinel/fusion bugs before
    # comparing the original full, non-factored relation and per-time graph math.
    reference.net.ri_lst = nn.ModuleList(reference.net.ri_lst)
    reference.net.gc_lst = nn.ModuleList(reference.net.gc_lst)
    reference.net.fs_lst = nn.ModuleList(reference.net.fs_lst)
    for relation in reference.net.ri_lst:
        relation.gc_lst = nn.ModuleList(relation.gc_lst)
    reference.net.glu = source.GLU(params['C']*(params['L']+1))
    reference.SRPE.srpe = nn.Embedding(params['alpha']+2,params['d'])
    reference.SRPE.SDist = current.hop_distance.clone()
    reference.range_mask = current.range_mask.float().repeat(1,params['beta']+1)
    return reference


@pytest.mark.parametrize('layers,beta,alpha,nodes,features,outputs,history',[
    (1,0,0,1,1,1,1),(1,2,1,4,2,1,4),(2,1,2,4,3,2,5),
    (2,5,1,3,2,2,3),(3,2,4,4,1,2,4),(3,0,0,1,2,2,2)])
def test_corrected_original_full_relation_forecast_and_every_gradient(layers,beta,alpha,nodes,features,outputs,history):
    torch.manual_seed(13)
    meta = metadata(nodes,features,outputs)
    params = dict(C=4,L=layers,d=3,alpha=alpha,beta=beta)
    current = STPGCN(metadata=meta,input_window=history,output_window=3,**params).double()
    old = corrected_reference(meta,current,params,history,3).double()
    loaded = current.load_state_dict({native_key(key):value for key,value in old.state_dict().items()},strict=False)
    assert loaded.missing_keys==['hop_distance','range_mask'] and not loaded.unexpected_keys
    positions = torch.stack((torch.arange(2*history).reshape(2,history)%7,
        torch.arange(2*history).reshape(2,history)*3%48),-1)
    source = torch.randn(2,history,nodes,features,dtype=torch.float64,requires_grad=True)
    reference_source = source.detach().clone().requires_grad_()
    actual = current(dict(X=source,history_features=positions))
    expected = old(reference_source,positions[...,0],positions[...,1])
    torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-12)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(source.grad,reference_source.grad,rtol=2e-9,atol=2e-12)
    reference = {native_key(key):value for key,value in old.named_parameters()}
    assert reference.keys()==dict(current.named_parameters()).keys()
    for key,value in current.named_parameters():
        assert value.grad is not None and reference[key].grad is not None
        torch.testing.assert_close(value.grad,reference[key].grad,rtol=2e-8,atol=2e-12)


def test_default_three_layers_widths_preserve_corrected_original_full_forward_and_gradients():
    torch.manual_seed(29)
    meta = metadata(features=1,outputs=1)
    params = dict(C=64,L=3,d=8,alpha=4,beta=2)
    current = STPGCN(metadata=meta,input_window=12,output_window=2)
    old = corrected_reference(meta,current,params,12,2)
    current.load_state_dict({native_key(key):value for key,value in old.state_dict().items()},strict=False)
    positions = torch.stack((torch.arange(12)%7,torch.arange(12)*2%48),-1)[None]
    source = torch.randn(1,12,4,1)
    actual = current(dict(X=source,history_features=positions))
    expected = old(source,positions[...,0],positions[...,1])
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    reference = {native_key(key):value for key,value in old.named_parameters()}
    for key,value in current.named_parameters():
        assert value.grad is not None and reference[key].grad is not None
        torch.testing.assert_close(value.grad,reference[key].grad,rtol=5e-4,atol=2e-6)


def test_factorized_weights_match_original_without_allocating_per_query_dense_graphs():
    torch.manual_seed(5)
    original = importlib.import_module('libcity.model.traffic_flow_prediction.STPGCN').STPRI(3,4,3,'cpu')
    original.gc_lst = nn.ModuleList(original.gc_lst)
    original.double()
    current = RelationInference(3,2).double()
    current.load_state_dict({key.replace('gc_lst','components'):value for key,value in original.state_dict().items()})
    spatial = torch.randn(4,3,dtype=torch.float64)*.3
    temporal = torch.randn(2,5,1,3,dtype=torch.float64)*.3
    relative_space = torch.randn(4,4,3,dtype=torch.float64)*.3
    relative_time = torch.randn(3,1,3,dtype=torch.float64)*.3
    mask = torch.tensor([[1,1,0,0],[0,1,1,0],[0,0,1,1],[0,0,0,1]],dtype=torch.bool)
    space,time = current(spatial,temporal,relative_space,relative_time,mask)
    assert space.shape==(4,4) and time.shape==(2,5,3)
    padded = torch.cat((temporal.new_zeros(2,2,1,3),temporal),1)
    for index in range(5):
        expected = original(spatial,temporal[:,index:index+1],padded[:,index:index+3],relative_space,relative_time)
        expected = expected*mask.repeat(1,3)
        actual = (time[:,index,:,None,None]*space[None,None]).permute(0,2,1,3).reshape(2,4,12)
        torch.testing.assert_close(actual,expected,rtol=2e-14,atol=1e-16)


@pytest.mark.parametrize('maximum',(0,1,2,4))
def test_shortest_outgoing_hops_match_independent_scipy_without_overwriting_distances(maximum):
    graph = np.array([[0,2,0,0,0],[0,0,.4,0,0],[0,0,1,0,0],[0,0,0,0,0],[1,0,0,0,0]],dtype=np.float32)
    original = graph.copy()
    graph.setflags(write=False)
    reference = shortest_path(graph>0,directed=True,unweighted=True)
    expected = np.where(np.isfinite(reference)&(reference<=maximum),reference,-1).astype(np.int64)
    actual = bounded_hops(graph,maximum)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(graph,original)
    assert actual[0,0]==0 and actual[0,4]==-1 and actual[3,3]==0


def test_hop_range_and_unreachable_embedding_are_separate_owned_static_buffers():
    meta = metadata()
    graph = meta.adjacency.copy()
    model = STPGCN(metadata=meta,input_window=4,output_window=2,C=4,L=2,d=2,alpha=2,beta=2).double()
    assert model.hop_distance[0,0]==0 and model.hop_distance[0,1]==1 and model.hop_distance[0,2]==2
    assert model.hop_distance[0,3]==3 and not model.range_mask[0,3]
    assert model.hop_distance[3,0]==3 and not model.range_mask[3,0]
    assert model.range_mask.dtype==torch.bool and model.hop_distance.dtype==torch.long
    assert set(dict(model.named_buffers()))=={'hop_distance','range_mask'}
    assert model.spatial_relative_embedding.num_embeddings==4
    np.testing.assert_array_equal(meta.adjacency,graph)
    assert all(not any(hasattr(part,key) for key in ('device','_device','_scaler','_logger')) for part in model.modules())
    parameters = dict(model.named_parameters())
    for index in range(2):
        assert all(f'layers.{index}.relation.components.{component}.{name}' in parameters
            for component in range(6) for name in ('mu','inv_sigma'))
        assert f'layers.{index}.convolution.x_proj.weight' in parameters
        assert f'layers.{index}.selection.fc.weight' in parameters


def test_temporal_graph_layers_are_causal_and_inputs_and_samples_are_independent():
    torch.manual_seed(17)
    model = STPGCN(metadata=metadata(),input_window=4,output_window=3,C=4,L=2,d=2,alpha=2,beta=2).double()
    source = torch.randn(2,4,4,2,dtype=torch.float64,requires_grad=True)
    positions = torch.tensor([[[6,46],[6,47],[0,0],[0,1]],[[2,5],[2,6],[2,7],[2,8]]])
    original = source.detach().clone()
    state = {key:value.clone() for key,value in model.state_dict().items()}
    predicted = model(dict(X=source,history_features=positions))
    torch.testing.assert_close(predicted[:1],model(dict(X=source[:1],history_features=positions[:1])),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(predicted,model(dict(X=source,history_features=positions,y=torch.full_like(predicted,float('nan')),
        future_features=source.new_full((2,3,4,5),float('nan')))),rtol=0,atol=0)
    predicted[0].sum().backward()
    torch.testing.assert_close(source.grad[1],torch.zeros_like(source.grad[1]),rtol=0,atol=0)
    spatial = model.spatial_embedding.weight
    temporal = (model.week_embedding(positions[...,0])+model.day_embedding(positions[...,1])).unsqueeze(2)
    relative_space = model.spatial_relative_embedding(model.hop_distance)
    relative_time = model.temporal_relative_embedding.weight[:,None]
    layer = model.layers[0]
    value = model.input_projection(source.detach())
    before = layer(value,spatial,temporal,relative_space,relative_time,model.range_mask)
    changed = value.clone()
    changed[:,2:] += 100
    changed_temporal = temporal.clone()
    changed_temporal[:,2:] += 20
    after = layer(changed,spatial,changed_temporal,relative_space,relative_time,model.range_mask)
    torch.testing.assert_close(before[:,:2],after[:,:2],rtol=0,atol=0)
    torch.testing.assert_close(source,original,rtol=0,atol=0)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,state[key],rtol=0,atol=0)


@pytest.mark.parametrize('interval',(60,300,900,1800,3600,86400))
def test_calendar_uses_actual_weekday_and_slot_across_midnight_without_float_rounding(interval):
    days = np.datetime64('2024-04-07T00:00:00')
    clock = days+np.arange(2*86400//interval)*np.timedelta64(interval,'s')
    actual = calendar_positions(clock,interval)
    points = 86400//interval
    np.testing.assert_array_equal(actual[:,0],np.repeat([6,0],points))
    np.testing.assert_array_equal(actual[:,1],np.tile(np.arange(points),2))
    assert actual.dtype==np.int64
    offset = calendar_positions(clock[:1]+np.timedelta64(interval//2,'s'),interval)
    assert offset[0,1]==0


def test_actual_clock_bundle_retains_common_samples_scaling_cache_and_target_isolation(tmp_path):
    folder = periodic_fixture(tmp_path/'raw',name='Tiny',steps=48,features=2,outputs=1,interval_minutes=5)
    path = folder/'Tiny.dyna'
    frame = pd.read_csv(path)
    clock = pd.date_range('2024-04-07T23:45:00Z',periods=48,freq='5min')
    frame['time'] = [clock[index%48].strftime('%Y-%m-%dT%H:%M:%SZ') for index in frame['dyna_id']]
    frame.to_csv(path,index=False)
    cfg = configuration(tmp_path,'calendar')
    cfg['training']['shuffle'] = False
    contexts = [make_context(cfg),make_context(cfg)]
    try:
        first,second = (context.bundle for context in contexts)
        assert first.split_summary['cache']['status']=='miss' and second.split_summary['cache']['status']=='hit'
        assert first.metadata.feature_dim==2 and first.metadata.input_layout['history_features']['steps_per_day']==288
        sample = first.train.dataset[0]
        expected = torch.tensor([[6,285],[6,286],[6,287],[0,0]],dtype=torch.long)
        torch.testing.assert_close(sample['history_features'],expected,rtol=0,atol=0)
        assert 'history_features' in model_inputs(sample,1)
        train,val,test = (getattr(first,name).dataset.windows for name in ('train','val','test'))
        assert train.target_starts.max()+1<val.target_starts.min()
        assert val.target_starts.max()+1<test.target_starts.min()
        np.testing.assert_allclose(first.scaler.mean,train.targets[:28].mean(),rtol=1e-6)
        before = val[0]
        before_calendar = first.val.dataset[0]['history_features'].clone()
        start = int(val.target_starts[0])
        frame.loc[(frame['dyna_id']%48)>=start,'value_0'] += 1000
        frame.to_csv(path,index=False)
        changed = make_context(cfg)
        try:
            after = changed.bundle.val.dataset[0]
            torch.testing.assert_close(after['X'],before['X'],rtol=0,atol=0)
            torch.testing.assert_close(after['history_features'],before_calendar,rtol=0,atol=0)
            assert not torch.equal(after['y_raw'],before['y_raw'])
            assert changed.bundle.split_summary['cache']['status']=='miss'
        finally:
            changed.tracker.close()
    finally:
        for context in contexts:
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


def test_full_registered_layer_train_evaluate_adam_schedule_rng_exact_resume(tmp_path):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=48,features=2,outputs=2)
    trained = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    partial = make_context(configuration(tmp_path,'partial'))
    initial = {key:value.clone() for key,value in partial.model.named_parameters()}
    train_loop(partial,until_epoch=1)
    gaussian = [key for key in initial if '.components.' in key]
    assert len(gaussian)==24 and any(not torch.equal(initial[key],dict(partial.model.named_parameters())[key]) for key in gaussian)
    for index in range(2):
        for component in ('relation','convolution','selection'):
            prefix = f'layers.{index}.{component}.'
            assert any(not torch.equal(value,dict(partial.model.named_parameters())[key])
                for key,value in initial.items() if key.startswith(prefix))
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    assert all(key in actual['model'] for key in gaussian)
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(2,4,2)


def test_single_node_tail_batch_cpu_amp_training(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=44,nodes=1)
    cfg = configuration(tmp_path,'amp')
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    context = make_context(cfg)
    try:
        assert [len(batch['X']) for batch in context.bundle.train]==[5,5,5,5,1]
    finally:
        context.tracker.close()
    cfg['runtime']['exp_id'] = 'amp-train'
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


def test_spawn_worker_can_collate_integer_calendar_history(tmp_path):
    atomic_fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'worker')
    cfg['training']['num_workers'] = 1
    context = make_context(cfg)
    try:
        batch = next(iter(context.bundle.train))
        assert batch['history_features'].dtype==torch.long and batch['history_features'].shape==(5,4,2)
    finally:
        context.tracker.close()


@pytest.mark.parametrize('interval',(0,-1,True,420,1.5))
def test_invalid_calendar_interval_is_rejected(interval):
    with pytest.raises(ValueError,match='time_intervals'):
        calendar_positions(np.array(['2024-01-01'],dtype='datetime64[s]'),interval)


@pytest.mark.parametrize('params',[dict(C=0),dict(L=0),dict(d=-1),dict(alpha=-1),dict(beta=-1),dict(beta=True)])
def test_invalid_model_structure_is_rejected(params):
    with pytest.raises(ValueError,match='STPGCN'):
        STPGCN(metadata=metadata(),input_window=4,output_window=2,**params)


def test_invalid_metadata_graph_and_prediction_inputs_fail_clearly():
    plain = DatasetMetadata('missing',1,1,1,adjacency=np.zeros((1,1)),input_layout=dict(kind='window'))
    with pytest.raises(ValueError,match='history_features'):
        STPGCN(metadata=plain,input_window=4,output_window=2)
    model = STPGCN(metadata=metadata(),input_window=4,output_window=2,C=4,L=1,d=2)
    source = torch.zeros(1,4,4,2)
    for positions in (None,torch.zeros(1,4,2),torch.zeros(1,3,2,dtype=torch.long)):
        with pytest.raises(ValueError,match='history_features'):
            model(dict(X=source,history_features=positions))
    positions = torch.zeros(1,4,2,dtype=torch.long)
    positions[0,0,0] = 7
    with pytest.raises(ValueError,match='calendar positions'):
        model(dict(X=source,history_features=positions))
    with pytest.raises(ValueError,match='X must'):
        model(dict(X=torch.zeros(1,3,4,2)))
