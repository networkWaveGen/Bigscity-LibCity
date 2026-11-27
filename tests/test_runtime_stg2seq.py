"""Corrected graph/attention axes and explicit autoregressive/teacher decoding."""
import copy
import json

import pandas as pd
import pytest
import torch

from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_demand_prediction.runtime_stg2seq import GraphTemporalConvolution,TemporalAttention,ChannelAttention
from libcity.model.traffic_flow_prediction.runtime_multiscale import spectral_basis
from libcity.model.traffic_speed_prediction.runtime_seq2seq import teacher_step
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,experiment):
    config = tiny_config(root,experiment,epochs=2,batch_size=8,shuffle=True)
    config['model'] = dict(name='STG2Seq',params=dict(window=2,dim_out=4,ks=2))
    config['training']['scheduler'] = dict(name='step',interval='step',step_size=2,gamma=.9)
    return config


@pytest.mark.parametrize('order',[1,3])
def test_graph_time_convolution_against_independent_per_axis_formula(order):
    source = torch.randn(2,2,4,3,dtype=torch.float64)
    graph = torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]]).numpy()
    basis = spectral_basis(graph,order).double()
    current = GraphTemporalConvolution(2,3,3,order).double()
    reference = copy.deepcopy(current)
    aligned = reference.align(source)
    batches = []
    for batch in range(2):
        times = []
        for time in range(4):
            nodes = []
            for destination in range(3):
                features = []
                for channel in range(2):
                    for offset in range(3):
                        for degree in range(order):
                            value = source.new_zeros(())
                            if time+offset<4:
                                for origin in range(3):
                                    value = value+source[batch,channel,time+offset,origin]*basis[degree,origin,destination]
                            features.append(value)
                logits = reference.weights@torch.stack(features)+reference.biases
                nodes.append((logits[:3]+aligned[batch,:,time,destination])*logits[3:].sigmoid())
            times.append(torch.stack(nodes,dim=-1))
        batches.append(torch.stack(times,dim=1))
    expected = torch.stack(batches)
    actual = current(source,basis)
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)
    actual.square().mean().backward();expected.square().mean().backward()
    for name,parameter in current.named_parameters():
        torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-8,atol=1e-8)


@pytest.mark.parametrize('kind',['time','channel'])
def test_attention_keeps_batch_and_node_axes_against_independent_formula(kind):
    source = torch.randn(2,3,4 if kind=='time' else 1,2,dtype=torch.float64)
    query = torch.randn(2,2,dtype=torch.float64)
    current = (TemporalAttention(4,2,3,2) if kind=='time' else ChannelAttention(2,3,2)).double()
    reference = copy.deepcopy(current)
    batches = []
    for batch in range(2):
        scores = []
        for position in range(4 if kind=='time' else 3):
            total = reference.bias[position]+query[batch]@reference.weight2[:,position]
            if kind=='time':
                for node in range(2):
                    for channel in range(3):
                        total = total+source[batch,channel,position,node]*reference.weight1[position,node*3+channel,0]
            else:
                for node in range(2):
                    total = total+source[batch,position,0,node]*reference.weight1[position,node,0]
            scores.append(total.tanh())
        weights = torch.stack(scores).softmax(0)
        if kind=='time':
            batches.append(sum(weights[t]*source[batch,:,t] for t in range(4))[:,None])
        else:
            batches.append(sum(weights[c]*source[batch,c,0] for c in range(3))[None,None])
    expected = torch.stack(batches)
    actual = current(query,source)
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)
    actual.square().mean().backward();expected.square().mean().backward()
    for name,parameter in current.named_parameters():
        torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-8,atol=1e-8)


def test_full_lifecycle_and_exact_teacher_and_batchnorm_resume(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=70)
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,teacher_step(partial.loss_fn),until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state']
    assert expected['scheduler']==actual['scheduler']
    for key,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)


@pytest.mark.parametrize('nodes',[1,3])
def test_multitarget_single_node_float64_and_label_input_isolation(tmp_path,nodes):
    folder = atomic_fixture(tmp_path/'raw',steps=70,nodes=nodes)
    frame = pd.read_csv(folder/'Tiny.dyna')
    frame['flow'],frame['occupancy'],frame['observation'] = frame['speed']*.5+2,frame['speed']*.1+1,frame['speed']*4
    frame.to_csv(folder/'Tiny.dyna',index=False)
    path = folder/'config.json';metadata = json.loads(path.read_text())
    metadata.update(data_col=['speed','flow','occupancy','observation'],output_dim=3)
    path.write_text(json.dumps(metadata))
    config = configuration(tmp_path,'inputs')
    if nodes==1:
        config['data']['features'] = dict(time_in_day=False)
        config['model']['params']['ks'] = 1
    context = make_context(config)
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        inputs = {key:value[:1].double() for key,value in model_inputs(batch,3).items()}
        original = inputs['X'].clone()
        states = {key:value.clone() for key,value in model.state_dict().items()}
        rng = torch.get_rng_state()
        prediction = model(inputs)
        assert prediction.shape==(1,2,nodes,3) and prediction.dtype==torch.float64
        assert torch.isfinite(prediction).all()
        torch.testing.assert_close(prediction,model({**inputs,'y':torch.full_like(prediction,float('nan'))}),rtol=0,atol=0)
        torch.testing.assert_close(original,inputs['X'],rtol=0,atol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key,value in states.items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in model.modules())
    finally:
        context.tracker.close()


def test_teacher_and_known_future_inputs_are_explicit(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=70)
    context = make_context(configuration(tmp_path,'teacher'))
    try:
        model = context.model.eval()
        batch = next(iter(context.bundle.test))
        inputs = model_inputs(batch,1)
        a = model.forward_with_teacher(inputs,torch.zeros_like(batch['y']))
        b = model.forward_with_teacher(inputs,torch.full_like(batch['y'],1000.))
        torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=0)
        assert not torch.allclose(a[:,1:],b[:,1:])
        with pytest.raises(ValueError,match='future_features'):
            model({'X':batch['X']})
    finally:
        context.tracker.close()


def test_actual_cpu_amp_training(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=70)
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0
