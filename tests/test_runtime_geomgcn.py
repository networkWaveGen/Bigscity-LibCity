"""Correct edge partitions against dense equations, then exercise real lifecycles."""
import copy
import itertools
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from libcity.data.bundle import DatasetMetadata
from libcity.model.road_representation.runtime_geomgcn import GeomGCN, GeometricHead
from libcity.executor.train import train_loop
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import road_feature_fixture
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root, experiment):
    return dict(data=dict(dataset='RoadFeatures', data_dir=str(root/'raw')),
        model=dict(name='GeomGCN', params=dict(hidden_dim=4, embedding_dim=3, dropout=.2,
            num_heads_layer_one=2, num_heads_layer_two=2)),
        training=dict(epochs=3, scheduler=dict(name='step', interval='epoch', step_size=1, gamma=.8)),
        runtime=dict(device='cpu', seed=17, output_dir=str(root/'runs'), exp_id=experiment))


def dense_partitions(metadata, dtype):
    # Independent full matrices are intentionally restricted to the tiny oracle.
    adjacency = torch.zeros(metadata.num_nodes, metadata.num_nodes, dtype=dtype)
    for source, destination in np.asarray(metadata.info['edge_index']).T:
        if source != destination:
            adjacency[destination, source] = 1
    own = torch.eye(metadata.num_nodes, dtype=dtype)
    # The reference DGL graph computes its degree normalization in float32.
    inverse = (adjacency + own).sum(1).float().rsqrt().to(dtype)
    return [inverse[:, None]*part*inverse[None, :] for part in (adjacency, own)]


def dense_layer(layer, features, partitions):
    # Reconstruct the original projection -> source norm -> incoming sum ->
    # merge -> destination norm -> activation, without calling runtime layers.
    channels = []
    for head in layer.heads:
        values = [part @ F.linear(features, projection.weight)
                  for part, projection in zip(partitions, head.projections)]
        value = torch.cat(values, -1) if head.merge == 'cat' else torch.stack(values).mean(0)
        channels.append(F.relu(value) if head.activation else value)
    return torch.cat(channels, -1) if layer.channel_merge == 'cat' else torch.stack(channels).mean(0)


@pytest.mark.parametrize('merges', list(itertools.product(('cat', 'mean'), repeat=4)))
def test_corrected_original_dense_forecast_embedding_and_all_gradients(tmp_path, merges, monkeypatch):
    road_feature_fixture(tmp_path/'raw')
    config = configuration(tmp_path, 'reference')
    keys = ('layer_one_ggcn_merge', 'layer_one_channel_merge', 'layer_two_ggcn_merge', 'layer_two_channel_merge')
    config['model']['params'].update(zip(keys, merges))
    context = make_context(config)
    try:
        current = context.model.eval().double()
        old = copy.deepcopy(current)
        partitions = dense_partitions(context.bundle.metadata, torch.float64)
        batch = next(iter(context.bundle.train))
        features = batch['node_features'].double()
        hidden = dense_layer(old.geomgcn1, features, partitions)
        embedding = dense_layer(old.geomgcn2, hidden, partitions)
        hidden = dense_layer(old.geomgcn3, embedding, partitions)
        expected = dense_layer(old.geomgcn4, hidden, partitions)[batch['node_indices']]
        monkeypatch.setattr(np, 'save', lambda *a, **k: pytest.fail('forward wrote a file'))
        actual = current(dict(node_features=features, node_indices=batch['node_indices']))
        torch.testing.assert_close(current.encode(features), embedding, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        assert actual.shape == batch['y'].shape
        width = 3*(2 if merges[2]=='cat' else 1)*(2 if merges[3]=='cat' else 1)
        assert embedding.shape == (12, width) and current.embedding_width == width
        actual.square().mean().backward()
        expected.square().mean().backward()
        reference = dict(old.named_parameters())
        for key, parameter in current.named_parameters():
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, reference[key].grad, rtol=1e-11, atol=1e-11)
    finally:
        context.tracker.close()


def test_edge_partitions_do_not_include_induced_edges_and_use_full_in_degree():
    # All nodes of partition 0 also occur in partition 1: induced subgraphs
    # would therefore mix both kinds of edges and cannot pass this oracle.
    edges = torch.tensor([[0, 0, 1], [1, 2, 2]])
    norm = torch.tensor([1., 2., 3., 1.]).rsqrt()[:, None]
    head = GeometricHead(1, 1, 0, 'cat', False).double()
    with torch.no_grad():
        head.projections[0].weight.fill_(2)
        head.projections[1].weight.fill_(3)
    features = torch.tensor([[1.], [2.], [3.], [7.]], dtype=torch.float64)
    norm = norm.double()
    output = head(features, edges, norm)
    expected = torch.zeros(4, 2, dtype=torch.float64)
    expected[:, 1] = 3*features[:, 0]*norm[:, 0].square()
    for source, destination in edges.T:
        expected[destination, 0] += 2*features[source, 0]*norm[source, 0]*norm[destination, 0]
    torch.testing.assert_close(output, expected, rtol=1e-14, atol=1e-14)
    assert output[-1, 0] == 0 and output[-1, 1] == 21


def assert_tensors_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_tensors_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        for left, right in zip(expected, actual, strict=True):
            assert_tensors_equal(left, right)
    else:
        assert expected == actual


def test_train_evaluate_export_and_exact_dropout_optimizer_rng_resume(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__
    def no_dgl(name, *args, **kwargs):
        if name == 'dgl' or name.startswith('dgl.'):
            pytest.fail('Native GeomGCN imported DGL')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_dgl)
    road_feature_fixture(tmp_path/'raw')
    full = run_experiment('train', config=configuration(tmp_path, 'full'))
    evaluated = run_experiment('evaluate', config=configuration(tmp_path, 'evaluate'), checkpoint=full.artifacts['best'])
    assert evaluated.metrics == {key:value for key,value in full.metrics.items() if key != 'best_validation'}
    assert np.load(evaluated.artifacts['embeddings'], allow_pickle=False).shape == (12, 3)
    assert json.loads(evaluated.artifacts['node_ids'].read_text()) == list(range(100, 112))
    info = json.loads(full.artifacts['runtime_info'].read_text())
    assert info['benchmark_protocol'] == 'road_reconstruction'
    assert info['split']['semantics'] == 'transductive_node_reconstruction'
    assert info['capabilities']['supports_resume'] is True
    partial = make_context(configuration(tmp_path, 'partial'))
    initial = {key:value.detach().clone() for key,value in partial.model.named_parameters()}
    train_loop(partial, until_epoch=1)
    changed = {key.split('.')[0] for key,value in partial.model.named_parameters()
               if not torch.equal(initial[key], value)}
    assert changed == {'geomgcn1','geomgcn2','geomgcn3','geomgcn4'}
    partial.tracker.close()
    resumed = run_experiment('train', config=configuration(tmp_path, 'resumed'),
        resume=partial.directory/'checkpoints/last.pt')
    expected = torch.load(full.artifacts['last'], weights_only=True)
    actual = torch.load(resumed.artifacts['last'], weights_only=True)
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        assert_tensors_equal(expected[key], actual[key])
    assert expected['train_state']['optimizer_steps'] == 3
    assert all(isinstance(value, torch.Tensor) for value in torch.load(full.artifacts['best'], weights_only=True).values())
    disabled = configuration(tmp_path, 'evaluate')
    disabled['evaluation'] = dict(save_predictions=False)
    result = run_experiment('evaluate', config=disabled, checkpoint=full.artifacts['best'])
    assert 'predictions' not in result.artifacts
    assert not (result.artifacts['embeddings'].parent/'predictions.npz').exists()


def test_label_free_dtype_stateless_buffers_and_full_vs_subset(tmp_path):
    road_feature_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path, 'dtype'))
    try:
        model = context.model.double().eval()
        source = next(iter(context.bundle.test))['node_features'].double()
        before = {key:value.clone() for key,value in model.state_dict().items()}
        original = source.clone()
        rng = torch.get_rng_state()
        predicted = model(dict(node_features=source))
        indices = torch.tensor([4])
        actual = model(dict(node_features=source, node_indices=indices, y=torch.full((1, 1), float('nan'))))
        assert actual.shape == (1, context.bundle.metadata.feature_dim) and actual.dtype == torch.float64
        torch.testing.assert_close(actual, predicted[indices], rtol=0, atol=0)
        torch.testing.assert_close(source, original, rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        assert_tensors_equal(before, model.state_dict())
        assert dict(model.named_buffers())['norm'].dtype == torch.float64
        assert not model.norm.requires_grad
        assert all(not any(hasattr(module, name) for name in ('device','_device','_scaler','_logger'))
                   for module in model.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('nodes', [1, 4])
def test_empty_graph_is_self_only_and_repeated_loops_do_not_change_degree(nodes):
    plain = DatasetMetadata('empty', nodes, 2, 2, info=dict(edge_index=np.empty((2, 0), dtype=np.int64)))
    repeated = np.tile(np.stack((np.arange(nodes), np.arange(nodes))), (1, 3))
    loops = DatasetMetadata('loops', nodes, 2, 2, info=dict(edge_index=repeated))
    model = GeomGCN(metadata=plain, hidden_dim=3, embedding_dim=2, dropout=0).double()
    loop_model = GeomGCN(metadata=loops, hidden_dim=3, embedding_dim=2, dropout=0).double()
    torch.testing.assert_close(loop_model.norm, model.norm, rtol=0, atol=0)
    assert loop_model.edge_index.shape == (2, 0)
    loop_model.load_state_dict(model.state_dict())
    torch.testing.assert_close(model.norm, torch.ones(nodes, 1, dtype=torch.float64))
    assert model.edge_index.shape == (2, 0)
    features = torch.arange(nodes*2, dtype=torch.float64).reshape(nodes, 2) + 1
    actual = model(dict(node_features=features))
    torch.testing.assert_close(actual, loop_model(dict(node_features=features)), rtol=0, atol=0)
    actual.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_sparse_two_thousand_node_metadata_and_actual_cpu_amp_training(tmp_path):
    road_feature_fixture(tmp_path/'raw', nodes=2000)
    context = make_context(configuration(tmp_path, 'sparse'))
    try:
        assert context.bundle.metadata.adjacency is None
        assert not context.bundle.metadata.info['edge_index'].flags.writeable
        assert context.model.edge_index.shape == (2, 4000)
        assert context.model.norm.shape == (2000, 1)
        assert sum(value.numel() for value in context.model.buffers()) == 10000
        batch = next(iter(context.bundle.val))
        actual = context.model(context.loss_fn.inputs(batch))
        assert actual.shape == batch['y'].shape and torch.isfinite(actual).all()
    finally:
        context.tracker.close()
    road_feature_fixture(tmp_path/'amp'/'raw')
    config = configuration(tmp_path/'amp', 'amp')
    config['training'].update(epochs=1, amp=True, amp_dtype='bfloat16')
    result = run_experiment('train', config=config)
    assert np.isfinite(list(result.metrics.values())).all()


def test_reconstruction_loss_preserves_double_precision_and_missing_target_gradients(tmp_path):
    road_feature_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path, 'loss'))
    try:
        prediction = torch.tensor([[1. + 1e-10, 10.]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[1., float('nan')]], dtype=torch.float64)
        result = context.loss_fn(prediction, target, target)
        assert result.weight == 1 and result.loss.dtype == torch.float64
        torch.testing.assert_close(result.loss, (prediction[0, 0]-1).square(), rtol=0, atol=0)
        assert result.loss > 0
        result.loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[2e-10, 0.]], dtype=torch.float64), atol=2e-17, rtol=0)
    finally:
        context.tracker.close()


@pytest.mark.parametrize('params', [dict(hidden_dim=0), dict(num_heads_layer_two=0), dict(dropout=1.),
                                 dict(layer_one_ggcn_merge='sum'), dict(layer_two_channel_merge='sum')])
def test_invalid_dimensions_and_merges_are_rejected(params):
    metadata = DatasetMetadata('tiny', 1, 1, 1, info=dict(edge_index=np.empty((2, 0), dtype=np.int64)))
    with pytest.raises(ValueError, match='GeomGCN'):
        GeomGCN(metadata=metadata, **params)
