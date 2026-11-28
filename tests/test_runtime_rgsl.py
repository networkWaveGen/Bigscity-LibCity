"""RGSL mixed-graph math, explicit sampling and full shared-runtime lifecycle."""
import importlib

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.executor.evaluate import evaluate_loop
from libcity.executor.rgsl_step import evaluate_rgsl,train
from libcity.model.runtime_graph import chebyshev_basis,scaled_laplacian
from libcity.model.traffic_flow_prediction.runtime_rgsl import RGSL,sample_edges
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,periodic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(nodes=4,features=2,outputs=2,graph=None):
    if graph is None:
        graph = np.roll(np.eye(nodes,dtype=np.float32),1,axis=1)
        if nodes>2:
            graph[0,2] = .5
    return DatasetMetadata('tiny',nodes,features,outputs,adjacency=graph,input_layout=dict(kind='window'))


def configuration(root,name):
    return dict(data=dict(dataset='Tiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=2,pad_training=False),
        model=dict(name='RGSL',params=dict(rnn_units=16,num_layers=2,embed_dim=4,cheb_k=3)),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
            scheduler=dict(name='exponential',interval='epoch',gamma=.8)),
        evaluation=dict(batch_size=2),runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


@pytest.mark.parametrize('units,layers,order,embedding,features,outputs,steps,horizon',[
    (64,2,2,8,1,1,12,12),(16,1,2,4,3,2,4,3),(16,2,3,5,2,2,3,2),(32,1,4,4,2,1,1,1)])
def test_original_mixed_graph_forecast_and_all_gradients_with_controlled_sampling(
        monkeypatch,units,layers,order,embedding,features,outputs,steps,horizon):
    original = importlib.import_module('libcity.model.traffic_flow_prediction.RGSL')
    meta = metadata(features=features,outputs=outputs)
    monkeypatch.setattr(original,'scaled_Laplacian',lambda graph:scaled_laplacian(graph).numpy())
    # Fix the legacy static Hadamard recurrence for K>2 in the reference.
    monkeypatch.setattr(original,'cheb_polynomial',lambda base,count:list(chebyshev_basis(torch.from_numpy(base),count).numpy()))
    torch.manual_seed(17)
    old = original.RGSL(dict(device=torch.device('cpu'),input_dim=features,output_dim=outputs,
        rnn_units=units,num_layers=layers,embed_dim=embedding,cheb_k=order,output_window=horizon),
        dict(num_nodes=4,adj_mx=meta.adjacency,scaler=NoneScaler())).double()
    for module in old.modules():
        if isinstance(module,original.AVWGCN):
            module.cheb_polynomials = [value.double() for value in module.cheb_polynomials]
    current = RGSL(metadata=meta,input_window=steps,output_window=horizon,rnn_units=units,
        num_layers=layers,embed_dim=embedding,cheb_k=order).double()
    loaded = current.load_state_dict(old.state_dict(),strict=False)
    assert not loaded.unexpected_keys and set(loaded.missing_keys)=={'static_basis','identity','off_diagonal'}
    noise = torch.randn(4,4,2,dtype=torch.float64,generator=torch.Generator().manual_seed(23))*.5
    def sampled(logits,tau=1,hard=False,**kwargs):
        scores = (logits+noise)/tau
        return F.one_hot(scores.argmax(-1),2).to(scores)-scores.detach()+scores
    monkeypatch.setattr(original,'gumbel_softmax',sampled)
    source = torch.randn(2,steps,4,features,dtype=torch.float64,requires_grad=True)
    reference_source = source.detach().clone().requires_grad_()
    expected = old(reference_source)
    actual,adjacency = current.forward_with_aux(dict(X=source),noise=noise)
    torch.testing.assert_close(adjacency,old.adj,rtol=0,atol=5e-16)
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=3e-10)
    actual.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(source.grad,reference_source.grad,rtol=2e-7,atol=3e-10)
    reference = dict(old.named_parameters())
    assert reference.keys()==dict(current.named_parameters()).keys()
    for name,value in current.named_parameters():
        assert value.grad is not None and reference[name].grad is not None
        torch.testing.assert_close(value.grad,reference[name].grad,rtol=4e-7,atol=5e-10)


def test_author_linear_straight_through_gradient_and_hard_values_are_preserved():
    logits = torch.tensor([[.7,.3],[.2,.8]],dtype=torch.float64,requires_grad=True)
    noise = torch.tensor([[-.5,.6],[.8,-.7]],dtype=torch.float64)
    actual = sample_edges(logits,.7,noise)
    torch.testing.assert_close(actual,torch.tensor([0.,1.],dtype=torch.float64),rtol=0,atol=0)
    actual.sum().backward()
    torch.testing.assert_close(logits.grad,torch.tensor([[1/.7,0],[1/.7,0]],dtype=torch.float64),rtol=0,atol=0)
    large = sample_edges(torch.tensor([[1e20,-1e20]]),noise=torch.zeros(1,2))
    assert large.item()==1


@pytest.mark.parametrize('order',(1,2,3,5))
def test_static_graph_basis_uses_matrix_chebyshev_recurrence(order):
    graph = np.array([[0,1,.5],[0,0,2],[1,0,0]],dtype=np.float32)
    base = scaled_laplacian(graph)
    actual = chebyshev_basis(base,order)
    expected = [torch.eye(3)]
    if order>1:
        expected.append(base)
    for _ in range(2,order):
        expected.append(2*base.matmul(expected[-1])-expected[-2])
    torch.testing.assert_close(actual,torch.stack(expected),rtol=0,atol=0)
    if order>2:
        assert not torch.allclose(actual[2],2*base*base-torch.eye(3))
    np.testing.assert_array_equal(graph,np.array([[0,1,.5],[0,0,2],[1,0,0]],dtype=np.float32))


@pytest.mark.parametrize('nodes,order,units',[(1,1,1),(1,3,2),(4,2,16),(4,4,3)])
def test_degenerate_graph_empty_sample_zero_embeddings_and_dtype_have_finite_gradients(nodes,order,units):
    model = RGSL(metadata=metadata(nodes=nodes,graph=np.zeros((nodes,nodes))),input_window=2,
        output_window=3,rnn_units=units,num_layers=1,embed_dim=2,cheb_k=order).double()
    with torch.no_grad():
        model.node_embeddings.zero_()
    source = torch.randn(1,2,nodes,2,dtype=torch.float64,requires_grad=True)
    # Force an empty learned adjacency, including the N>1 case.
    noise = torch.stack((-torch.ones(nodes,nodes),torch.ones(nodes,nodes)),-1).double()*10
    prediction,adjacency = model.forward_with_aux(dict(X=source),noise=noise)
    assert prediction.shape==(1,3,nodes,2) and not adjacency.any()
    assert torch.isfinite(prediction).all() and prediction.dtype==torch.float64
    prediction.square().mean().backward()
    assert torch.isfinite(source.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    torch.testing.assert_close(model.static_basis[0],torch.eye(nodes,dtype=torch.float64))
    if order>1:
        torch.testing.assert_close(model.static_basis[1],-torch.eye(nodes,dtype=torch.float64))


def test_forward_is_label_free_deterministic_batch_independent_and_does_not_cache_graphs():
    model = RGSL(metadata=metadata(),input_window=4,output_window=2,rnn_units=16,num_layers=2).double()
    source = torch.randn(2,4,4,2,dtype=torch.float64,requires_grad=True)
    before = {key:value.clone() for key,value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    first = model(dict(X=source,y=torch.full((1,),float('nan'))))
    model.eval()
    repeated = model(dict(X=source))
    separate = torch.cat([model(dict(X=value[None])) for value in source],0)
    torch.testing.assert_close(first,repeated,rtol=0,atol=0)
    torch.testing.assert_close(first,separate,rtol=2e-13,atol=2e-13)
    first[0].sum().backward()
    assert source.grad[1].abs().sum()==0 and source.grad[0].abs().sum()>0
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert not any(hasattr(model,key) for key in ('adj','tilde','device','_scaler','_logger'))
    assert all(buffer.device==source.device for buffer in model.buffers())


def test_graph_and_embedding_parameters_are_prepared_once_per_forward(monkeypatch):
    model = RGSL(metadata=metadata(),input_window=7,output_window=2,rnn_units=16,num_layers=2,cheb_k=4)
    calls = []
    for cell in model.encoder.dcrnn_cells:
        for block in (cell.gate,cell.update):
            original = block.prepare
            def counted(embeddings,original=original):
                calls.append(1)
                return original(embeddings)
            monkeypatch.setattr(block,'prepare',counted)
    basis_calls = []
    module = importlib.import_module('libcity.model.traffic_flow_prediction.runtime_rgsl')
    original_basis = module.chebyshev_basis
    def counted_basis(base,order):
        basis_calls.append(1)
        return original_basis(base,order)
    monkeypatch.setattr(module,'chebyshev_basis',counted_basis)
    model(dict(X=torch.randn(2,7,4,2)))
    assert len(calls)==4 and len(basis_calls)==1


def test_evaluation_uses_one_fresh_learned_basis_per_call(tmp_path,monkeypatch):
    atomic_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'evaluate'))
    try:
        arguments = dict(device=context.device,loss_fn=context.loss_fn,evaluation=context.config['evaluation'])
        expected = evaluate_loop(context.model,context.bundle.test,**arguments).metrics
        calls = []
        original = context.model.inference_basis
        def counted():
            calls.append(1)
            return original()
        monkeypatch.setattr(context.model,'inference_basis',counted)
        assert evaluate_rgsl(context.model,context.bundle.test,**arguments).metrics==expected and len(calls)==1
        with torch.no_grad():
            context.model.node_embeddings[0].neg_()
        calls.clear()
        actual = evaluate_rgsl(context.model,context.bundle.test,**arguments).metrics
        assert len(calls)==1
        assert actual==evaluate_loop(context.model,context.bundle.test,**arguments).metrics
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


def test_real_atomic_train_evaluate_and_stochastic_adam_schedule_rng_exact_resume(tmp_path):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=60,nodes=4,features=2,outputs=2)
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    partial = make_context(configuration(tmp_path,'partial'))
    before = partial.model.node_embeddings.detach().clone()
    train(partial,until_epoch=1)
    assert not torch.equal(before,partial.model.node_embeddings)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    assert len([key for key in actual['model'] if 'weights_pool' in key])==4
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(2,4,2)


@pytest.mark.parametrize('nodes,order,units',[(1,1,1),(1,3,2),(4,3,16)])
def test_actual_cpu_amp_with_true_single_sample_tail(tmp_path,nodes,order,units):
    atomic_fixture(tmp_path/'raw',steps=50,nodes=nodes)
    cfg = configuration(tmp_path,'amp')
    cfg['data'].update(input_window=4,output_window=1)
    cfg['model']['params'].update(rnn_units=units,num_layers=1,cheb_k=order)
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


@pytest.mark.parametrize('params',[dict(rnn_units=0),dict(num_layers=0),dict(embed_dim=0),dict(cheb_k=0)])
def test_invalid_dimensions_are_rejected(params):
    with pytest.raises(ValueError,match='RGSL'):
        RGSL(metadata=metadata(),input_window=4,output_window=2,**params)


@pytest.mark.parametrize('strategy',[dict(temperature=0),dict(temperature=float('inf')),dict(unknown=1)])
def test_invalid_training_policy_is_rejected(strategy):
    with pytest.raises(ValueError):
        load_config(config=dict(model=dict(name='RGSL'),data=dict(dataset='Tiny'),training=dict(strategy=strategy)))


def test_invalid_graph_and_input_dimensions_are_rejected():
    with pytest.raises(ValueError,match='square adjacency'):
        RGSL(metadata=metadata(graph=np.ones((2,3))),input_window=4,output_window=2)
    with pytest.raises(ValueError,match='match metadata'):
        RGSL(metadata=metadata(graph=np.ones((2,2))),input_window=4,output_window=2)
    model = RGSL(metadata=metadata(),input_window=4,output_window=2,rnn_units=16)
    with pytest.raises(ValueError,match='X must'):
        model(dict(X=torch.zeros(1,3,4,2)))
