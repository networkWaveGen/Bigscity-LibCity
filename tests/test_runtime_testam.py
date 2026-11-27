"""TESTAM retains expert math and route supervision in the shared lifecycle."""
import json
import math

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config
from libcity.executor.components import build_scheduler
from libcity.executor.evaluate import model_inputs
from libcity.executor.testam_step import expert_step,routing_labels
from libcity.executor.train import train_loop
from libcity.model.traffic_speed_prediction.TESTAM import TESTAM as ReferenceTESTAM
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,experiment,quantile=True):
    config = tiny_config(root,experiment,epochs=3,batch_size=8,shuffle=True)
    config['model'] = dict(name='TESTAM',params=dict(hidden_size=8,layers=1,dropout=.1,is_quantile=quantile))
    config['training']['scheduler'] = dict(name='inverse_sqrt_cosine',interval='step',model_dim=8,warmup_steps=4)
    return config


def reference_experts(model,inputs,horizon):
    """Keep old expert blocks; explicitly remove its residual from forecast output."""
    source = inputs['X'].transpose(1,3)
    current = (source[:,-1,0]*model.vocab_size).round().long()
    future = (current+horizon)%model.vocab_size
    first,second = model.gate_network.We1@model.gate_network.memory,model.gate_network.We2@model.gate_network.memory
    supports = [F.softmax(F.relu(first@second.T),-1),F.softmax(F.relu(second@first.T),-1)]
    identity,identity_states = model.identity_expert(current,source[:,:1])
    _,future_states = model.identity_expert(future)
    _,adaptive,adaptive_states = model.adaptive_expert(source,future_states,supports)
    attention,attention_state = model.attention_expert(source,future_states)
    experts = torch.stack((identity,adaptive[...,-1:],attention),dim=-2)
    gate = F.softmax(model.gate_network(source.permute(0,2,3,1),
        [identity_states[-1],adaptive_states[-1],attention_state]),-1)
    result = torch.zeros_like(identity)
    # Reference routing selects the expert axis, not flattened feature positions.
    for expert in range(3):
        result = torch.where((gate.argmax(-1)==expert)[...,None],experts[...,expert,:],result)
    if model.prob_mul:
        result = result*gate.max(-1).values[...,None]
    return result[:,:,-horizon:].permute(0,2,1,3),gate[:,:,-horizon:],experts[:,:,-horizon:]


@pytest.mark.parametrize('probability_weighted',[False,True])
def test_expert_reference_auxiliary_loss_and_gradients(tmp_path,probability_weighted):
    atomic_fixture(tmp_path/'raw',steps=80)
    config = configuration(tmp_path,'reference')
    config['model']['params']['dropout'] = 0.
    config['model']['params']['prob_mul'] = probability_weighted
    config['data'].update(input_window=12,output_window=12)
    context = make_context(config)
    try:
        reference = ReferenceTESTAM(dict(hidden_size=8,layers=1,dropout=0.,output_dim=1,time_intervals=300,prob_mul=probability_weighted),
            dict(num_nodes=3,feature_dim=2,ext_dim=1,scaler=NoneScaler()))
        context.model.load_state_dict(reference.state_dict())
        batch = next(iter(context.bundle.train))
        # All targets valid: compare the original quantile objective exactly.
        batch['y_raw'] = batch['y'] = torch.full_like(batch['y'],7.)
        expected,gate,experts = reference_experts(reference,model_inputs(batch,1),12)
        actual = context.model.forward_with_aux(model_inputs(batch,1))
        for result,wanted in zip(actual,(expected,gate,experts)):
            torch.testing.assert_close(result,wanted,rtol=1e-5,atol=1e-6)
        error = (experts-batch['y'].permute(0,2,1,3).unsqueeze(-2)).abs()
        routed = (expected-batch['y']).abs().permute(0,2,1,3)
        chosen = F.one_hot(gate.argmax(-1),3).float()
        wrong = routed>routed.quantile(.7)
        correct = routed<routed.quantile(.3)
        worst = torch.where(wrong,(1-chosen)/2,chosen)
        best = chosen*correct
        expected_loss = error.mean()+(-.5*(worst+best)*gate.log()).mean()
        result = expert_step(context.loss_fn)(context.model,batch,context.state)
        torch.testing.assert_close(result.loss,expected_loss)
        assert result.weight==batch['y'].numel()
        expected_loss.backward()
        result.loss.backward()
        for name,parameter in context.model.named_parameters():
            torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-5,atol=1e-6)
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in context.model.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('quantile',[True,False])
def test_complete_training_independent_evaluation_and_exact_resume(tmp_path,quantile):
    atomic_fixture(tmp_path/'raw',steps=60)
    full = run_experiment('train',config=configuration(tmp_path,'full',quantile))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate',quantile),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial',quantile))
    train_loop(partial,expert_step(partial.loss_fn),until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume',quantile),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state']
    assert expected['scheduler']==actual['scheduler']
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)


def test_routing_labels_use_expert_axis_and_ignore_invalid_positions():
    gate = torch.tensor([[[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.1,.1,.8]]]])
    errors = torch.tensor([[[[3.,2.,1.],[1.,3.,2.],[2.,3.,1.],[3.,1.,2.]]]])
    valid = torch.tensor([[[True,True,True,False]]])
    worst,best = routing_labels(gate,errors,torch.ones_like(valid,dtype=torch.float),valid)
    torch.testing.assert_close(worst[0,0,:3],torch.tensor([[0.,.5,.5],[.5,0.,.5],[0.,0.,1.]]))
    torch.testing.assert_close(best[0,0,:3],torch.tensor([[.25,.25,.5],[.5,.25,.25],[0.,0.,1.]]))
    assert not worst[...,3,:].any() and not best[...,3,:].any()
    invalid = torch.zeros_like(valid)
    assert all(not value.any() for value in routing_labels(gate,errors,errors[...,0],invalid,quantile=.7))


def test_raw_mask_multitarget_calendar_and_label_free_dtype(tmp_path):
    folder = atomic_fixture(tmp_path/'raw',steps=60)
    frame = pd.read_csv(folder/'Tiny.dyna')
    frame['flow'],frame['occupancy'] = frame['speed']*.5+3,frame['speed']*.01
    extra = [f'observation{index}' for index in range(6)]
    for index,column in enumerate(extra):
        frame[column] = frame['speed']*(index+1)*.1
    frame.to_csv(folder/'Tiny.dyna',index=False)
    metadata = json.loads((folder/'config.json').read_text())
    metadata.update(data_col=['speed','flow','occupancy',*extra],output_dim=2)
    (folder/'config.json').write_text(json.dumps(metadata))
    config = configuration(tmp_path,'inputs')
    config['model']['params']['dropout'] = 0.
    context = make_context(config)
    try:
        assert context.model.time_index==9 and context.model.observed==tuple(range(9))
        batch = next(iter(context.bundle.test))
        inputs = {key:value[:1].double() for key,value in model_inputs(batch,2).items()}
        current = context.model.train().double()
        before = torch.get_rng_state()
        prediction = current(inputs)
        assert prediction.shape==(1,2,3,2) and prediction.dtype==torch.float64
        torch.testing.assert_close(prediction,current({**inputs,'y':torch.full_like(prediction,float('nan'))}),rtol=0,atol=0)
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
        torch.testing.assert_close(prediction,current({'X':inputs['X']}),rtol=1e-5,atol=1e-6)
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in current.parameters())
        current.float()
        batch['y_raw'].fill_(0.)
        batch['y_raw'][0,0,0,0] = 5.
        batch['y_raw'][0,0,0,1] = float('nan')
        result = expert_step(context.loss_fn)(current,batch,context.state)
        assert result.weight==1 and torch.isfinite(result.loss)
        batch['y_raw'].zero_()
        empty = expert_step(context.loss_fn)(current,batch,context.state)
        assert empty.weight==0 and empty.loss==0 and empty.loss.requires_grad
    finally:
        context.tracker.close()


def test_original_learning_rate_curve_and_safe_resumable_state():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter],lr=.001)
    config = dict(name='inverse_sqrt_cosine',interval='step',model_dim=8,warmup_steps=4,lr_multiplier=2.)
    schedule = build_scheduler(optimizer,config,epochs=3,steps_per_epoch=4)
    for update in range(1,14):
        expected = 2*8**-.5*(min(update**-.5,update*4**-1.5) if update<=4
            else 4**-.5*(1+math.cos(math.pi*((update-4)%4)/4)))
        assert optimizer.param_groups[0]['lr']==pytest.approx(expected)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        schedule.step('step')


def test_cpu_amp_and_input_guards(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0
    config = configuration(tmp_path,'horizon')
    config['data']['output_window'] = 5
    with pytest.raises(ValueError,match='output_window'):
        make_context(config)
    config = configuration(tmp_path,'calendar')
    config['data']['features'] = {'time_in_day':False}
    with pytest.raises(ValueError,match='time_in_day'):
        make_context(config)
    config = configuration(tmp_path,'schedule')
    config['training']['scheduler']['warmup_steps'] = 0
    with pytest.raises(ValueError,match='warmup_steps'):
        load_config(config=config)
