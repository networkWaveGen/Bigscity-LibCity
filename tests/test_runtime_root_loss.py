"""Nonlinear RMSE reduction must preserve full-batch gradients and metrics."""
import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from libcity.executor.components import build_optimizer,build_scheduler
from libcity.executor.evaluate import aggregate_loss,evaluate_loop
from libcity.executor.root_loss import RootMeanSquaredLoss
from libcity.executor.train import train_loop
from libcity.testing.fixtures import atomic_fixture
from libcity.utils.normalization import StandardScaler,NoneScaler
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def root_loss(name='masked_rmse',mask=0,scaler=None):
    bundle = SimpleNamespace(metadata=SimpleNamespace(output_dim=1),scaler=scaler or NoneScaler())
    return RootMeanSquaredLoss(bundle,dict(name=name,mask_value=mask))


def root_context(config):
    context = make_context(config)
    # Exercise the public custom-loss injection before CCRNN is registered.
    config = dict(name='masked_rmse',mask_value=0)
    context.config['training']['loss'] = config
    context.loss_fn = RootMeanSquaredLoss(context.bundle,config)
    return context


@pytest.mark.parametrize('name,mask',[('rmse',0),('rmse',None),('masked_rmse',0),('masked_rmse',None)])
def test_raw_scale_additive_statistic_and_gradient_equal_direct_rmse(name,mask):
    loss = root_loss(name,mask,StandardScaler(10.,3.))
    prediction = torch.tensor([.1,.3,.7,1.,1.1],dtype=torch.float64).reshape(5,1,1,1).requires_grad_()
    raw_target = torch.tensor([0.,9.,11.,float('nan'),13.],dtype=torch.float64).reshape_as(prediction)
    target = (raw_target-10)/3
    result = loss(prediction,target,raw_target)
    independent = prediction.detach().clone().requires_grad_()
    valid = torch.isfinite(raw_target)
    if name.startswith('masked_') and mask is not None:
        valid &= raw_target!=mask
    residual = (3*independent+10)[valid]-raw_target[valid]
    expected = residual.square().mean().sqrt()
    total = result.loss.item()*result.weight
    assert result.weight==int(valid.sum())
    assert aggregate_loss(loss,total,result.weight)==pytest.approx(expected.item(),rel=1e-14)
    (result.loss*result.weight).backward()
    prediction.grad.div_(loss.gradient_divisor(total,result.weight))
    expected.backward()
    torch.testing.assert_close(prediction.grad,independent.grad,rtol=1e-13,atol=1e-14)


def test_exact_fit_has_zero_finite_gradient_and_nonempty_update_divisor():
    loss = root_loss()
    prediction = torch.ones(4,1,1,1,requires_grad=True)
    result = loss(prediction,torch.ones_like(prediction))
    total = result.loss.item()*result.weight
    assert aggregate_loss(loss,total,result.weight)==0
    assert loss.gradient_divisor(total,result.weight)==4
    (result.loss*result.weight).backward()
    assert torch.equal(prediction.grad,torch.zeros_like(prediction))


def test_completely_masked_batch_keeps_zero_differentiable_statistics():
    loss = root_loss()
    prediction = torch.tensor([1.,2.],requires_grad=True).reshape(2,1,1,1)
    result = loss(prediction,torch.tensor([0.,float('nan')]).reshape_as(prediction))
    assert result.weight==0 and aggregate_loss(loss,result.loss.item(),0)==0
    assert loss.gradient_divisor(0.,0)==0
    result.loss.backward()
    assert result.loss.item()==0


class PredictionInputs(torch.nn.Module):
    def forward(self,inputs):
        return inputs['X']


@pytest.mark.parametrize('batch_size',(1,2,4,7))
def test_evaluation_root_is_global_and_independent_of_missing_tail_partition(tmp_path,batch_size):
    prediction = torch.tensor([1.,3.,2.,4.,7.,5.,9.]).reshape(7,1,1,1)
    target = torch.tensor([2.,2.,0.,float('nan'),3.,9.,10.]).reshape_as(prediction)
    dataset = [dict(X=a,y=b,y_raw=b) for a,b in zip(prediction,target,strict=True)]
    loader = DataLoader(dataset,batch_size=batch_size)
    loss = root_loss()
    result = evaluate_loop(PredictionInputs(),loader,device=torch.device('cpu'),loss_fn=loss,
        evaluation=dict(metrics=['MAE','RMSE'],mask_value=0.,mode='single',save_predictions=True),directory=tmp_path)
    valid = torch.isfinite(target)&(target!=0)
    expected = float(torch.sqrt(((prediction[valid].double()-target[valid])**2).mean()))
    assert result.metrics['loss']==pytest.approx(expected,rel=2e-8)
    assert result.metrics['RMSE@1']==pytest.approx(expected,rel=2e-8)
    with np.load(result.artifacts['predictions'],allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['prediction'],prediction.numpy())


@pytest.mark.parametrize('clip',(None,.7))
def test_accumulation_and_short_group_match_independent_full_batch_root_updates(tmp_path,clip):
    atomic_fixture(tmp_path/'raw')
    cfg = tiny_config(tmp_path,'accumulate',epochs=1,batch_size=2,accumulation_steps=3,
        optimizer=dict(name='sgd',lr=.02,momentum=.3,weight_decay=.01),
        scheduler=dict(name='step',interval='step',step_size=1,gamma=.8),clip_grad_norm=clip)
    context = root_context(cfg)
    reference = copy.deepcopy(context.model)
    optimizer = build_optimizer(reference,context.config['training']['optimizer'])
    loader = DataLoader(context.bundle.train.dataset,batch_size=6,shuffle=False)
    schedule = build_scheduler(optimizer,context.config['training']['scheduler'],epochs=1,steps_per_epoch=len(loader))
    total,count = 0.,0
    try:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = reference(context.loss_fn.inputs(batch))
            raw_prediction,raw_target = context.loss_fn.raw(prediction,batch['y'],batch['y_raw'])
            valid = torch.isfinite(raw_target)&(raw_target!=0)
            residual = raw_prediction[valid]-raw_target[valid]
            objective = residual.square().mean().sqrt()
            total += residual.square().sum().item()
            count += int(valid.sum())
            objective.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(reference.parameters(),clip)
            optimizer.step()
            schedule.step('step')
        train_loop(context)
        assert context.state.optimizer_steps==len(loader)==4
        for name,value in context.model.state_dict().items():
            torch.testing.assert_close(value,reference.state_dict()[name],rtol=5e-6,atol=5e-7)
        for value,expected in zip(context.optimizer.state.values(),optimizer.state.values(),strict=True):
            torch.testing.assert_close(value['momentum_buffer'],expected['momentum_buffer'],rtol=5e-6,atol=5e-7)
        assert context.schedule.state_dict()==schedule.state_dict()
        metrics = json.loads((context.directory/'metrics.jsonl').read_text().splitlines()[-1])
        assert metrics['train_loss']==pytest.approx(math.sqrt(total/count),rel=2e-6)
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


def test_root_objective_adam_metric_schedule_shuffle_and_rng_resume_exactly(tmp_path):
    atomic_fixture(tmp_path/'raw')
    options = dict(epochs=3,batch_size=5,accumulation_steps=2,shuffle=True,
        scheduler=dict(name='plateau',interval='metric',monitor='val_loss',patience=0,factor=.5))
    def create(name):
        cfg = tiny_config(tmp_path,name,**options)
        cfg['model']['params'].update(num_layers=2,dropout=.3)
        return root_context(cfg)
    full = create('full')
    contexts = [full]
    try:
        train_loop(full)
        partial = create('partial')
        contexts.append(partial)
        train_loop(partial,until_epoch=1)
        resumed = create('resumed')
        contexts.append(resumed)
        resumed.checkpoints.load_last(partial.directory/'checkpoints/last.pt',resumed)
        train_loop(resumed)
        expected,actual = (torch.load(context.directory/'checkpoints/last.pt',weights_only=True) for context in (full,resumed))
        for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
            equal(expected[key],actual[key])
    finally:
        for context in contexts:
            context.tracker.close()


def test_empty_root_groups_do_not_advance_optimizer_or_step_schedule(tmp_path):
    atomic_fixture(tmp_path/'raw')
    context = root_context(tiny_config(tmp_path,'empty',epochs=1,batch_size=4,
        scheduler=dict(name='step',interval='step',step_size=1,gamma=.5)))
    try:
        def empty(model,batch,state):
            return context.loss_fn(model(context.loss_fn.inputs(batch)),torch.zeros_like(batch['y']))
        initial = {key:value.clone() for key,value in context.model.state_dict().items()}
        train_loop(context,empty)
        assert context.state.optimizer_steps==0 and context.optimizer.param_groups[0]['lr']==.001
        for key,value in context.model.state_dict().items():
            torch.testing.assert_close(value,initial[key],rtol=0,atol=0)
    finally:
        context.tracker.close()


def test_root_actual_cpu_amp_and_short_accumulation_group(tmp_path):
    atomic_fixture(tmp_path/'raw')
    context = root_context(tiny_config(tmp_path,'amp',epochs=1,batch_size=5,accumulation_steps=2,
        amp=True,amp_dtype='bfloat16'))
    try:
        assert np.isfinite(train_loop(context))
        assert context.state.optimizer_steps==3
    finally:
        context.tracker.close()


def test_root_loss_requires_explicit_objective_name():
    with pytest.raises(ValueError,match='requires rmse'):
        root_loss('mse')
