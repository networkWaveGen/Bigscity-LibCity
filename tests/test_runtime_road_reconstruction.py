"""Road objectives keep graph reconstruction separate from LINE link sampling."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from libcity.config.runtime import load_config
from libcity.executor.train import train_loop
from libcity.model.road_representation.runtime_reconstruction import ChebyshevLayer, graph_matrix
from libcity.model.utils import get_supports_matrix
from libcity.pipeline.experiment import run_experiment, select_device
from libcity.testing.fixtures import road_feature_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,name,experiment):
    parameters = dict(embedding_dim=4,max_diffusion_step=1) if name=='ChebConv' else dict(embedding_dim=4,dropout=.2)
    return dict(data=dict(dataset='RoadFeatures',data_dir=str(root/'raw')),
        model=dict(name=name,params=parameters),training=dict(epochs=3),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


@pytest.mark.parametrize('name',['ChebConv','GAT'])
def test_reference_forward_and_gradients_with_no_forward_filesystem(tmp_path,name,monkeypatch):
    road_feature_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,name,'reference'))
    try:
        model = context.model.eval()
        meta = context.bundle.metadata
        legacy = getattr(importlib.import_module('libcity.model.road_representation.'+name),name)
        config = dict(device=torch.device('cpu'),output_dim=4,max_diffusion_step=1,filter_type='dual_random_walk')
        old = legacy(config,dict(adj_mx=graph_matrix(meta),num_nodes=meta.num_nodes,
            feature_dim=meta.feature_dim,scaler=NoneScaler())).eval()
        loaded = model.load_state_dict(old.state_dict(),strict=False)
        assert not loaded.unexpected_keys
        assert all('support_' in key or key=='edge_index' for key in loaded.missing_keys)
        batch = next(iter(context.bundle.train))
        if name=='ChebConv':
            expected = old.decoder(old.encoder(batch['node_features']))[batch['node_indices']]
        else:
            encoded = old.encoder([batch['node_features'],old.Apt])[0]
            expected = old.decoder([encoded,old.Apt])[0][batch['node_indices']]
        monkeypatch.setattr(np,'save',lambda *a,**k:pytest.fail('Model forward wrote an embedding file'))
        actual = model(context.loss_fn.inputs(batch))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for key,value in model.named_parameters():
            torch.testing.assert_close(value.grad,dict(old.named_parameters())[key].grad,rtol=1e-5,atol=1e-6)
        assert all(not any(hasattr(module,key) for key in ('device','_device','_scaler','_logger')) for module in model.modules())
    finally:
        context.tracker.close()


def test_chebyshev_zero_order_and_independent_dual_support_polynomials():
    adjacency = np.array([[0,1,0],[0,0,1],[1,1,0]],dtype=np.float32)
    source = torch.tensor([[1.,2.],[3.,5.],[7.,11.]])
    zero = ChebyshevLayer(adjacency,2,3,0,'dual_random_walk')
    torch.testing.assert_close(zero(source),source@zero.weight+zero.biases)
    layer = ChebyshevLayer(adjacency,2,3,2,'dual_random_walk')
    supports = [torch.tensor(matrix.toarray(),dtype=source.dtype) for matrix in get_supports_matrix(adjacency,'dual_random_walk')]
    expected_terms = [source]
    for support in supports:
        expected_terms.extend((support@source,(2*support@support-torch.eye(3))@source))
    expected = torch.stack(expected_terms,dim=-1).flatten(1)@layer.weight+layer.biases
    torch.testing.assert_close(layer(source),expected)


@pytest.mark.parametrize('name',['ChebConv','GAT'])
def test_full_lifecycle_exact_resume_protocol_and_embeddings(tmp_path,name):
    road_feature_fixture(tmp_path/'raw')
    full = run_experiment('train',config=configuration(tmp_path,name,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,name,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    info = json.loads(full.artifacts['runtime_info'].read_text())
    assert info['benchmark_protocol']=='road_reconstruction'
    assert info['protocol_changes']['benchmark.protocol']=='road_reconstruction'
    assert info['split']['semantics']=='transductive_node_reconstruction'
    assert np.load(evaluated.artifacts['embeddings'],allow_pickle=False).shape==(12,4)
    assert json.loads(evaluated.artifacts['node_ids'].read_text())==list(range(100,112))
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,name,'partial'))
    train_loop(partial,until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,name,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state']
    for key,value in expected['model'].items():
        if value.is_sparse:
            value,value2=value.to_dense(),actual['model'][key].to_dense()
        else:
            value2=actual['model'][key]
        torch.testing.assert_close(value,value2,rtol=0,atol=0)


def test_node_splits_scaling_missing_values_and_data_ownership(tmp_path):
    folder = road_feature_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'GAT','data')
    first = make_context(config)
    try:
        bundle = first.bundle
        partition = bundle.split_summary['node_indices']
        assert set(partition['train']).isdisjoint(partition['val'])
        assert set(partition['train']).isdisjoint(partition['test'])
        assert set(partition['val']).isdisjoint(partition['test'])
        assert set(sum(partition.values(),[]))==set(range(12))
        previous = bundle.scaler.mean.copy()
        batch = next(iter(bundle.train))
        batch['node_features'].zero_()
        assert torch.count_nonzero(next(iter(bundle.train))['node_features'])>0
        frame = pd.read_csv(folder/'RoadFeatures.geo')
        frame.loc[partition['val']+partition['test'],'length'] += 1e6
        frame.loc[partition['val'][0],'width'] = np.nan
        frame.to_csv(folder/'RoadFeatures.geo',index=False)
        second = make_context(configuration(tmp_path,'GAT','changed'))
        try:
            np.testing.assert_array_equal(second.bundle.scaler.mean,previous)
            validation = next(iter(second.bundle.val))
            assert torch.isfinite(validation['node_features']).all()
            assert torch.isnan(validation['y_raw']).any()
            prediction = second.model(second.loss_fn.inputs(validation))
            result = second.loss_fn(prediction,validation['y'],validation['y_raw'])
            assert result.weight==int(torch.isfinite(validation['y_raw']).sum())
        finally:
            second.tracker.close()
    finally:
        first.tracker.close()


@pytest.mark.parametrize('name',['ChebConv','GAT'])
def test_float64_subset_prediction_and_cpu_amp(tmp_path,name):
    road_feature_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,name,'dtype'))
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        inputs = context.loss_fn.inputs(batch)
        inputs['node_features'] = inputs['node_features'].double()
        inputs['node_indices'] = inputs['node_indices'][:1]
        prediction = model(inputs)
        assert prediction.shape==(1,context.bundle.metadata.feature_dim) and prediction.dtype==torch.float64
        torch.testing.assert_close(prediction,model({**inputs,'y':torch.full_like(prediction,float('nan'))}))
    finally:
        context.tracker.close()
    config = configuration(tmp_path,name,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0


def test_gat_empty_graph_and_protocol_metric_validation(tmp_path):
    folder = road_feature_fixture(tmp_path/'raw')
    frame = pd.read_csv(folder/'RoadFeatures.rel')
    frame.iloc[:0].to_csv(folder/'RoadFeatures.rel',index=False)
    result = run_experiment('train',config=configuration(tmp_path,'GAT','empty-graph'))
    assert np.isfinite(list(result.metrics.values())).all()
    config = configuration(tmp_path,'GAT','wrong-metric')
    config['evaluation'] = dict(metrics=['accuracy'])
    with pytest.raises(ValueError,match='Unknown evaluation metric'):
        load_config(config=config)


@pytest.mark.parametrize('name',['ChebConv','GAT'])
def test_sparse_road_metadata_and_model_do_not_expand_dense_adjacency(tmp_path,name):
    road_feature_fixture(tmp_path/'raw',nodes=2000)
    context = make_context(configuration(tmp_path,name,'sparse'))
    try:
        metadata = context.bundle.metadata
        assert metadata.adjacency is None
        assert metadata.info['edge_index'].shape==(2,4000)
        assert not metadata.info['edge_index'].flags.writeable
        context.model.eval()
        batch = next(iter(context.bundle.val))
        prediction = context.model(context.loss_fn.inputs(batch))
        assert prediction.shape==batch['y'].shape and torch.isfinite(prediction).all()
        if name=='ChebConv':
            supports = [value for key,value in context.model.named_buffers() if 'support_' in key]
            assert supports and all(value.is_sparse and value._nnz()==4000 for value in supports)
        else:
            assert dict(context.model.named_buffers())['edge_index'].shape==(2,4000)
    finally:
        context.tracker.close()


def test_automatic_device_selection_respects_model_capabilities(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    assert str(select_device('auto',allowed=('cpu','cuda'))) == 'cpu'
    assert str(select_device('auto',allowed=('cpu','mps'))) == 'mps'
    assert str(select_device('auto',allowed=('cpu',))) == 'cpu'
    with pytest.raises(ValueError,match='No available device'):
        select_device('auto',allowed=('cuda',))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    assert str(select_device('auto',allowed=('cpu','cuda'))) == 'cuda:0'
