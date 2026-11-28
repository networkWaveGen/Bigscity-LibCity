"""DKFN retains dual-stream equations while removing batch and label coupling."""
import importlib
import itertools

import numpy as np
import pytest
import torch

from libcity.data.bundle import DatasetMetadata
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_speed_prediction.runtime_dkfn import DKFN,fuse_streams,sample_variance
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import periodic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,experiment):
    return dict(data=dict(dataset='PeriodicTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=2,pad_training=False),
        model=dict(name='DKFN',params=dict(hops=2)),
        training=dict(epochs=3,batch_size=8,scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        evaluation=dict(batch_size=17),runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


def reference_one_step(old,source):
    size = len(source)
    width = old.output_dim*old.num_nodes
    h,c,rh,rc = (source.new_zeros(size,width) for _ in range(4))
    for index in range(source.shape[1]):
        value = source[:,index].transpose(1,2).reshape(size,-1)
        filtered = torch.cat([layer(value) for layer in old.gc_list],1)
        combined = torch.cat((filtered,h),1)
        neighbor = old.A_list[-1].repeat(old.output_dim,old.output_dim)@old.Neighbor_weight
        c = torch.sigmoid(old.fl(combined))*c*neighbor+torch.sigmoid(old.il(combined))*torch.tanh(old.Cl(combined))
        h = torch.sigmoid(old.ol(combined))*torch.tanh(c)
        combined = torch.cat((value,rh),1)
        rc = torch.sigmoid(old.rfl(combined))*rc+torch.sigmoid(old.ril(combined))*torch.tanh(old.rCl(combined))
        rh = torch.sigmoid(old.rol(combined))*torch.tanh(rc)
        a = value.var(-1,keepdim=True,correction=int(value.shape[-1]>1))
        b = filtered.var(-1,keepdim=True,correction=int(filtered.shape[-1]>1))
        prediction = (h*a*old.c+rh*b)/(a+b*old.c).clamp_min(torch.finfo(source.dtype).tiny)
        prediction = torch.where((a+b)>0,prediction,(h+rh)/2)
    return prediction.reshape(size,old.output_dim,old.num_nodes).transpose(1,2)


@pytest.mark.parametrize('hops,clamp,dimensions',list(itertools.product((1,3),(False,True),((1,1),(2,1),(2,2)))))
def test_original_recurrent_equations_with_explicit_local_variances_and_positive_coefficient_gradients(tmp_path,hops,clamp,dimensions):
    features,outputs = dimensions
    periodic_fixture(tmp_path/'raw',steps=60,features=features,outputs=outputs)
    cfg = configuration(tmp_path,'reference')
    cfg['model']['params'].update(hops=hops,clamp_adjacency=clamp)
    context = make_context(cfg)
    try:
        current = context.model.eval()
        metadata = context.bundle.metadata
        old_type = importlib.import_module('libcity.model.traffic_speed_prediction.DKFN').DKFN
        old = old_type(dict(K_hop_numbers=hops,Clamp_A=clamp,device=torch.device('cpu')),
            dict(num_nodes=4,feature_dim=features,output_dim=outputs,adj_mx=metadata.adjacency.copy(),scaler=NoneScaler())).eval()
        incoming = old.state_dict()
        incoming['c'] = torch.log(torch.expm1(old.c.detach()))
        loaded = current.load_state_dict(incoming,strict=False)
        assert not loaded.unexpected_keys
        assert all(key in ('supports','cell_filter') or key.endswith('.filter_square_matrix') for key in loaded.missing_keys)
        batch = {key:value[:2] for key,value in next(iter(context.bundle.train)).items()}
        source = batch['X']
        expected = []
        for _ in range(2):
            predicted = reference_one_step(old,source)
            expected.append(predicted)
            following = source[:,-1].clone()
            following[...,:outputs] = predicted
            source = torch.cat((source[:,1:],following[:,None]),1)
        expected = torch.stack(expected,1)
        actual = current(model_inputs(batch,outputs))
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        reference = dict(old.named_parameters())
        for key,value in current.named_parameters():
            assert value.grad is not None
            gradient = reference[key].grad
            if key=='c':
                gradient = gradient*torch.sigmoid(current.c.detach())
            torch.testing.assert_close(value.grad,gradient,rtol=3e-4,atol=2e-6)
    finally:
        context.tracker.close()


def test_variance_fusion_matches_author_formula_and_has_neutral_zero_variance_fallback():
    graph = torch.tensor([[.2,-.5]],dtype=torch.float64,requires_grad=True)
    direct = torch.tensor([[-.1,.3]],dtype=torch.float64,requires_grad=True)
    observed = torch.tensor([[1.,3.,5.]],dtype=torch.float64,requires_grad=True)
    filtered = torch.tensor([[2.,4.]],dtype=torch.float64,requires_grad=True)
    coefficient = torch.tensor([.3],dtype=torch.float64,requires_grad=True)
    actual = fuse_streams(graph,direct,observed,filtered,coefficient)
    expected = (graph*4*.3+direct*2)/(4+2*.3)
    torch.testing.assert_close(actual,expected,rtol=1e-14,atol=1e-14)
    actual.sum().backward()
    for value in (graph,direct,observed,filtered,coefficient):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    zero = torch.ones(1,1,dtype=torch.float64,requires_grad=True)
    fallback = fuse_streams(graph,direct,zero,zero,coefficient)
    torch.testing.assert_close(fallback,(graph+direct)/2,rtol=0,atol=0)
    assert sample_variance(zero).item()==0


def test_batch_neighbors_do_not_change_prediction_or_receive_cross_sample_gradients(tmp_path):
    periodic_fixture(tmp_path/'raw',steps=60)
    context = make_context(configuration(tmp_path,'independence'))
    try:
        model = context.model.eval().double()
        source = next(iter(context.bundle.test))['X'][:2].double()
        single = model(dict(X=source[:1]))
        mixed = source.clone()
        mixed[1] = mixed[1]*100+30
        predicted = model(dict(X=mixed))
        torch.testing.assert_close(single,predicted[:1],rtol=1e-12,atol=1e-12)
        mixed.requires_grad_()
        model(dict(X=mixed))[0].sum().backward()
        torch.testing.assert_close(mixed.grad[1],torch.zeros_like(mixed.grad[1]),rtol=0,atol=0)
        model.train()
        torch.testing.assert_close(single,model(dict(X=source[:1])),rtol=0,atol=0)
    finally:
        context.tracker.close()


def test_graph_powers_clamping_and_static_buffer_dtype_do_not_mutate_metadata():
    graph = np.array([[1,1,1,1],[0,1,0,0],[0,1,1,0],[0,1,0,1]],dtype=np.float32)
    metadata = DatasetMetadata('graph',4,1,1,adjacency=graph,input_layout=dict(kind='window'))
    original = metadata.adjacency.copy()
    plain = DKFN(metadata=metadata,input_window=2,output_window=1,hops=3)
    clipped = DKFN(metadata=metadata,input_window=2,output_window=1,hops=3,clamp_adjacency=True)
    assert plain.supports[1,0,1]>1 and clipped.supports.max()<=1
    np.testing.assert_array_equal(metadata.adjacency,original)
    value = torch.tensor(graph)
    normalized = torch.diag(value.sum(0).reciprocal())@value
    expected = torch.stack([torch.linalg.matrix_power(normalized,index) for index in range(1,4)])
    torch.testing.assert_close(plain.supports,expected)
    plain.double()
    assert all(value.dtype==torch.float64 for value in plain.buffers())
    assert all(not any(hasattr(part,key) for key in ('device','_device','_logger','_scaler')) for part in plain.modules())


@pytest.mark.parametrize('nodes,hops,constant',[(1,1,0.),(1,3,2.),(4,1,0.),(4,3,2.)])
def test_scalar_constant_isolated_graphs_and_coefficient_extremes_have_finite_gradients(nodes,hops,constant):
    metadata = DatasetMetadata('degenerate',nodes,1,1,adjacency=np.zeros((nodes,nodes)),input_layout=dict(kind='window'))
    model = DKFN(metadata=metadata,input_window=3,output_window=2,hops=hops).double()
    for coefficient in (-40.,0.,40.):
        with torch.no_grad():
            model.c.fill_(coefficient)
        model.zero_grad(set_to_none=True)
        source = torch.full((1,3,nodes,1),constant,dtype=torch.float64,requires_grad=True)
        prediction = model(dict(X=source))
        assert prediction.shape==(1,2,nodes,1) and torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        assert torch.isfinite(source.grad).all()
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())


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


def test_full_train_evaluate_and_exact_rmsprop_schedule_rng_resume(tmp_path):
    periodic_fixture(tmp_path/'raw',steps=60)
    trained = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluated'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    partial = make_context(configuration(tmp_path,'partial'))
    initial = {key:value.detach().clone() for key,value in partial.model.named_parameters()}
    train_loop(partial,until_epoch=1)
    assert any(not torch.equal(initial[key],value) for key,value in partial.model.named_parameters())
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(2,4,2)


def test_multistep_uses_only_declared_known_future_features_and_retains_unknown_observations(tmp_path):
    periodic_fixture(tmp_path/'raw',steps=60,features=2,outputs=1)
    cfg = configuration(tmp_path,'future')
    cfg['data']['features'] = dict(time_in_day=True)
    cfg['data']['output_window'] = 3
    context = make_context(cfg)
    try:
        model = context.model.eval().double()
        batch = {key:value[:1] for key,value in next(iter(context.bundle.test)).items()}
        inputs = {key:value.double() for key,value in model_inputs(batch,1).items()}
        original = {key:value.clone() for key,value in inputs.items()}
        state = {key:value.clone() for key,value in model.state_dict().items()}
        predicted = model(inputs)
        altered = {key:value.clone() for key,value in inputs.items()}
        altered['future_features'][:,0] += 2
        actual = model(altered)
        torch.testing.assert_close(predicted[:,0],actual[:,0],rtol=0,atol=0)
        assert not torch.allclose(predicted[:,1:],actual[:,1:])
        torch.testing.assert_close(model({**inputs,'y':torch.full_like(predicted,float('nan'))}),predicted,rtol=0,atol=0)
        equal(inputs,original)
        equal(state,model.state_dict())
        with pytest.raises(ValueError,match='future_features'):
            model(dict(X=inputs['X']))
        # The final future frame is not needed to predict any subsequent step.
        altered = {key:value.clone() for key,value in inputs.items()}
        altered['future_features'][:,-1] += 1e5
        torch.testing.assert_close(predicted,model(altered),rtol=0,atol=0)
    finally:
        context.tracker.close()


def test_actual_cpu_amp_training(tmp_path):
    periodic_fixture(tmp_path/'raw',steps=60)
    cfg = configuration(tmp_path,'amp')
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert np.isfinite(list(run_experiment('train',config=cfg).metrics.values())).all()


@pytest.mark.parametrize('params',[dict(hops=0),dict(threshold=-1.),dict(threshold=float('nan')),dict(clamp_adjacency=1)])
def test_invalid_model_parameters_are_rejected(params):
    metadata = DatasetMetadata('tiny',1,1,1,adjacency=np.zeros((1,1)),input_layout=dict(kind='window'))
    with pytest.raises(ValueError,match='DKFN'):
        DKFN(metadata=metadata,input_window=2,output_window=1,**params)
