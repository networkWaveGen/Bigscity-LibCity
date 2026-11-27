"""Graph sequence decoding and nodewise temporal attention use standard inputs."""
import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.runtime_registry import model_spec,resolve
from libcity.model.traffic_flow_prediction.ToGCN import ToGCN as ReferenceToGCN
from libcity.model.traffic_flow_prediction.SimST import SimST as ReferenceSimST,TemporalAttention
from libcity.model.traffic_flow_prediction.runtime_simst import neighbor_projection
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')

PARAMS = {'ToGCN':dict(hidden_size=4,decoder_t=2,teacher_forcing_ratio=.5),
          'SimST':dict(node_dim=3,init_dim=4,end_dim=8,layer=1,dropout=.1,in_neighbor_num=1)}


def configuration(root,name,experiment):
    config = tiny_config(root,experiment,epochs=2,batch_size=8,shuffle=True)
    config['model'] = dict(name=name,params=dict(PARAMS[name]))
    config['training']['scheduler'] = dict(name='step',interval='epoch',step_size=1,gamma=.8)
    return config


@pytest.mark.parametrize('teacher',[False,True])
def test_togcn_original_forecast_teacher_objective_and_gradients(tmp_path,monkeypatch,teacher):
    atomic_fixture(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,'ToGCN','reference')
    config['model']['params']['teacher_forcing_ratio'] = float(teacher)
    context = make_context(config)
    try:
        anomaly = torch.is_anomaly_enabled()
        # The reference constructor's global anomaly side effect is not math.
        monkeypatch.setattr(torch.autograd,'set_detect_anomaly',lambda *_: None)
        metadata = context.bundle.metadata
        reference = ReferenceToGCN(dict(hidden_size=4,decoder_t=2,teacher_forcing_ratio=float(teacher)),
            dict(num_nodes=3,feature_dim=1,output_dim=1,adj_mx=metadata.adjacency.copy(),scaler=NoneScaler())).eval()
        current = context.model.eval()
        missing = current.load_state_dict(reference.state_dict(),strict=False)
        assert missing.missing_keys==['adjacency'] and not missing.unexpected_keys
        batch = next(iter(context.bundle.train))
        expected = reference.predict(batch)
        actual = current.forward_with_teacher(model_inputs(batch,1),batch['y']) if teacher else current(model_inputs(batch,1))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for name,parameter in current.named_parameters():
            torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-5,atol=1e-6)
        assert torch.is_anomaly_enabled()==anomaly
    finally:
        context.tracker.close()


def reference_augmented(source,adjacency,highlighted):
    """Explicit per-node construction independent of the vectorized adapter."""
    # Fixture channels are speed, time of day, seven weekday flags.
    base = torch.cat((source[...,:2],source[...,2:9].argmax(-1,keepdim=True).to(source.dtype)/7),-1)
    rows = []
    for node in range(len(adjacency)):
        selections = []
        top = []
        all_neighbors = []
        for graph in (adjacency,adjacency.T):
            neighbors = [i for i in range(len(graph)) if i!=node and graph[node,i]!=0]
            ranked = sorted(neighbors,key=lambda i:(graph[node,i],i))[-highlighted:] if highlighted else []
            ranked += [node]*max(0,highlighted-len(ranked))
            top.append(ranked)
            all_neighbors.append(neighbors or [node])
        for position in range(highlighted-1,-1,-1):
            selections.extend([source[:,:,top[0][position],0],source[:,:,top[1][position],0]])
        selections.extend(source[:,:,indices,0].mean(-1) for indices in all_neighbors)
        rows.append(torch.stack(selections,-1))
    return torch.cat((base,torch.stack(rows,2)),dim=-1)


@pytest.mark.parametrize('highlighted',[0,1])
def test_simst_original_attention_forecast_and_gradients(tmp_path,highlighted):
    atomic_fixture(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,'SimST','reference')
    config['data']['output_window'] = 4
    config['model']['params']['in_neighbor_num'] = highlighted
    context = make_context(config)
    try:
        metadata = context.bundle.metadata
        current = context.model.eval()
        reference = ReferenceSimST(dict(node_dim=3,in_dim=5+2*highlighted,init_dim=4,end_dim=8,layer=1,
            dropout=.1,output_window=4,output_dim=1),dict(num_nodes=3,scaler=NoneScaler())).eval()
        missing = current.load_state_dict(reference.state_dict(),strict=False)
        assert missing.missing_keys==['neighbor_projection'] and not missing.unexpected_keys
        batch = next(iter(context.bundle.train))
        packed = reference_augmented(batch['X'],metadata.adjacency,highlighted)
        torch.testing.assert_close(current.packed_inputs(batch['X']),packed)
        expected = reference.predict(dict(X=packed,node_idx=torch.zeros(packed.shape[0],dtype=torch.long)))
        actual = current(model_inputs(batch,1))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for name,parameter in current.named_parameters():
            torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-5,atol=1e-6)
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name',PARAMS)
def test_full_training_independent_evaluation_and_exact_resume(tmp_path,name):
    atomic_fixture(tmp_path/'raw',steps=60)
    full = run_experiment('train',config=configuration(tmp_path,name,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,name,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,name,'partial'))
    step = model_spec(name).step
    train_loop(partial,resolve(step)(partial.loss_fn) if step else None,until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,name,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state']
    assert expected['scheduler']==actual['scheduler']
    assert expected['rng']['python']==actual['rng']['python']
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)


@pytest.mark.parametrize('name',PARAMS)
def test_multi_target_batch_one_float64_input_ownership_and_label_isolation(tmp_path,name):
    folder = atomic_fixture(tmp_path/'raw',steps=60)
    frame = pd.read_csv(folder/'Tiny.dyna')
    frame['flow'] = frame['speed']*.5+2
    frame['observation'] = frame['speed']*.1
    frame.to_csv(folder/'Tiny.dyna',index=False)
    metadata = json.loads((folder/'config.json').read_text())
    metadata.update(data_col=['speed','flow','observation'],output_dim=2)
    (folder/'config.json').write_text(json.dumps(metadata))
    context = make_context(configuration(tmp_path,name,'dtype'))
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        inputs = {key:value[:1].double() for key,value in model_inputs(batch,2).items()}
        original = {key:value.clone() for key,value in inputs.items()}
        python_rng = random.getstate()
        torch_rng = torch.get_rng_state()
        prediction = model(inputs)
        assert prediction.shape==(1,2,3,2) and prediction.dtype==torch.float64
        assert torch.isfinite(prediction).all()
        torch.testing.assert_close(prediction,model({**inputs,'y':torch.full_like(prediction,float('nan'))}),rtol=0,atol=0)
        assert random.getstate()==python_rng
        torch.testing.assert_close(torch.get_rng_state(),torch_rng,rtol=0,atol=0)
        for key,value in original.items():
            torch.testing.assert_close(value,inputs[key],rtol=0,atol=0)
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in model.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name',PARAMS)
def test_actual_cpu_amp_training(tmp_path,name):
    atomic_fixture(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,name,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0


@pytest.mark.parametrize('node_dimension',[0,3])
def test_simst_large_batch_has_same_node_semantics_and_causal_buffer(tmp_path,node_dimension):
    atomic_fixture(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,'SimST','large')
    config['model']['params']['node_dim'] = node_dimension
    context = make_context(config)
    try:
        model = context.model.eval()
        batch = next(iter(context.bundle.test))
        source = batch['X'][:1]
        expected = model({'X':source})
        actual = model({'X':source.expand(65,-1,-1,-1)})
        torch.testing.assert_close(actual,expected.expand(65,-1,-1,-1),rtol=1e-5,atol=1e-6)
        for key,value in model.named_buffers():
            if key.endswith('mask'):
                assert value.dtype==torch.bool and value.shape==(4,4)
        attention = TemporalAttention(4,heads=2,window_size=4).eval()
        tokens = torch.randn(2,4,4)
        changed = tokens.clone();changed[:,-1] += 20
        torch.testing.assert_close(attention(tokens)[:,:-1],attention(changed)[:,:-1],rtol=0,atol=0)
    finally:
        context.tracker.close()


def test_neighbor_projection_is_directional_normalized_and_handles_isolated_nodes():
    adjacency = np.array([[0.,3.,1.],[0.,0.,0.],[2.,0.,0.]])
    before = adjacency.copy()
    projection = neighbor_projection(adjacency,2)
    np.testing.assert_array_equal(adjacency,before)
    np.testing.assert_allclose(projection.sum(-1),1.)
    np.testing.assert_array_equal(projection[0,0],[0.,1.,0.])
    np.testing.assert_array_equal(projection[-2,0],[0.,.5,.5])
    np.testing.assert_array_equal(projection[-1,0],[0.,0.,1.])
    np.testing.assert_array_equal(projection[-2,1],[0.,1.,0.])


def test_togcn_teacher_is_explicit_and_constructor_has_no_global_side_effect(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,'ToGCN','teacher')
    config['model']['params']['teacher_forcing_ratio'] = 1.
    config['data']['features'] = dict(time_in_day=True)
    before = torch.is_anomaly_enabled()
    context = make_context(config)
    try:
        assert torch.is_anomaly_enabled()==before
        model = context.model.eval()
        batch = next(iter(context.bundle.test))
        inputs = model_inputs(batch,1)
        first = model.forward_with_teacher(inputs,torch.zeros_like(batch['y']))
        second = model.forward_with_teacher(inputs,torch.full_like(batch['y'],30.))
        torch.testing.assert_close(first[:,0],second[:,0],rtol=0,atol=0)
        assert not torch.allclose(first[:,1:],second[:,1:])
        with pytest.raises(ValueError,match='future features'):
            model.forward_with_teacher({'X':batch['X']},batch['y'])
    finally:
        context.tracker.close()
    config['model']['params']['decoder_t'] = 4
    with pytest.raises(ValueError,match='decoder_t'):
        make_context(config)
