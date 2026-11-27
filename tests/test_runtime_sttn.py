"""Stateless graph transforms, device-safe layers and precise shared resume."""
import json
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_speed_prediction.STTN import STTN as ReferenceSTTN
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,experiment):
    config = tiny_config(root,experiment,epochs=2,batch_size=8,shuffle=True)
    config['model'] = dict(name='STTN',params=dict(embed_dim=4,num_layers=2,num_heads=2,
        forward_expansion=2,dropout_rate=.1))
    config['training']['scheduler'] = dict(name='step',interval='step',step_size=2,gamma=.9)
    return config


def first_spatial_reference(block,value,key,query):
    """Original first-call graph formula, independent of the new spatial forward."""
    count,nodes,steps,width = query.shape
    graph = block.adj_mx
    normalized = (graph-graph.mean())/torch.sqrt(graph.var(unbiased=False)+block.norm_adj.eps)
    graph_states = torch.stack([block.gcn(query[:,:,position],normalized) for position in range(steps)],dim=2)
    spatial = block.embed_linear(block.D_S)[None,:,None].expand(count,nodes,steps,width)
    query = query+spatial
    attention = block.attention(value,key,query)
    residual = block.dropout_layer(block.norm1(attention+query))
    transformed = block.dropout_layer(block.norm2(block.feed_forward(residual)+residual))
    gate = torch.sigmoid(block.fs(transformed)+block.fg(graph_states))
    return gate*transformed+(1-gate)*graph_states


def test_original_first_forecast_and_gradients(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=60)
    context = make_context(configuration(tmp_path,'reference'))
    try:
        metadata = context.bundle.metadata
        # Graph normalization is the original first-call formula, applied to
        # the immutable source graph; repeated normalization is intentionally fixed.
        reference = ReferenceSTTN(dict(embed_dim=4,num_layers=2,num_heads=2,forward_expansion=2,
            dropout_rate=.1,input_window=4,output_window=2,TG_in_one_day=288),
            dict(feature_dim=1,output_dim=1,adj_mx=metadata.adjacency.copy(),scaler=NoneScaler())).eval()
        for block in reference.transformer.encoder.layers:
            block.STransformer.forward = partial(first_spatial_reference,block.STransformer)
        model = context.model.eval()
        model.load_state_dict(reference.state_dict())
        batch = next(iter(context.bundle.train))
        expected = reference.predict(batch)
        actual = model(model_inputs(batch,1))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-5,atol=1e-6)
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in model.modules())
    finally:
        context.tracker.close()


def test_full_training_independent_evaluation_and_dropout_exact_resume(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=60)
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state']
    assert expected['scheduler']==actual['scheduler']
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)


@pytest.mark.parametrize('nodes',[1,3])
def test_static_graph_multitarget_float64_label_and_rng_isolation(tmp_path,nodes):
    folder = atomic_fixture(tmp_path/'raw',steps=60,nodes=nodes,interval_minutes=60)
    frame = pd.read_csv(folder/'Tiny.dyna');frame['flow'] = frame['speed']*.5+2
    frame.to_csv(folder/'Tiny.dyna',index=False)
    path = folder/'config.json';metadata = json.loads(path.read_text())
    metadata.update(data_col=['speed','flow'],output_dim=2)
    path.write_text(json.dumps(metadata))
    context = make_context(configuration(tmp_path,'state'))
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        inputs = {key:value[:1].double() for key,value in model_inputs(batch,2).items()}
        original = {name:value.clone() for name,value in model.state_dict().items()}
        source = inputs['X'].clone()
        rng = torch.get_rng_state()
        prediction = model(inputs)
        assert prediction.shape==(1,2,nodes,2) and prediction.dtype==torch.float64
        assert torch.isfinite(prediction).all()
        for _ in range(3):
            torch.testing.assert_close(prediction,model({**inputs,'y':torch.full_like(prediction,float('nan'))}),rtol=0,atol=0)
        torch.testing.assert_close(source,inputs['X'],rtol=0,atol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for name,value in original.items():
            torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        for block in model.transformer.encoder.layers:
            graph = block.STransformer.adj_mx
            assert graph.dtype==torch.float64
            np.testing.assert_array_equal(graph.cpu().numpy(),context.bundle.metadata.adjacency)
            assert block.TTransformer.temporal_embedding.num_embeddings==24
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        context.tracker.close()


def test_cpu_amp_training_and_invalid_position_size(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0
    config = configuration(tmp_path,'position')
    config['model']['params']['position_size'] = 2
    with pytest.raises(ValueError,match='position_size'):
        make_context(config)
