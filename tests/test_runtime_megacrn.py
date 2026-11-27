"""Memory auxiliaries, explicit curricula and deterministic label-free inference."""
import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.runtime_registry import model_spec,resolve
from libcity.model.traffic_speed_prediction.MegaCRN import MegaCRN as ReferenceMegaCRN
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,experiment):
    config = tiny_config(root,experiment,epochs=3,batch_size=8,shuffle=True)
    config['data']['output_window'] = 3
    config['model'] = dict(name='MegaCRN',params=dict(rnn_units=4,num_layers=2,cheb_k=3,mem_num=5,
        mem_dim=3,cl_decay_steps=3,lamb=.02,lamb1=.03))
    config['training']['scheduler'] = dict(name='step',interval='step',step_size=2,gamma=.9)
    return config


def test_reference_prediction_auxiliaries_objective_and_gradients(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=80)
    config = configuration(tmp_path,'reference')
    config['model']['params']['use_curriculum_learning'] = False
    context = make_context(config)
    try:
        metadata = context.bundle.metadata
        reference = ReferenceMegaCRN(dict(context.config['model']['params'],output_window=3,device=torch.device('cpu')),
            dict(num_nodes=3,output_dim=1,ext_dim=1,scaler=NoneScaler()))
        context.model.load_state_dict(reference.state_dict())
        batch = next(iter(context.bundle.train))
        expected = reference(batch['X'][...,:1],batch['future_features'],labels=batch['y'],batches_seen=0)
        actual = context.model.forward_with_aux(model_inputs(batch,1),targets=batch['y'],batches_seen=0)
        for value,other in zip(expected,actual):
            torch.testing.assert_close(value,other,rtol=1e-5,atol=1e-6)
        prediction,_,query,positive,negative = expected
        main = context.loss_fn(prediction,batch['y'],batch['y_raw']).loss
        expected_loss = main+.02*F.triplet_margin_loss(query,positive.detach(),negative.detach())+.03*F.mse_loss(query,positive.detach())
        actual_loss = resolve(model_spec('MegaCRN').step)(context.loss_fn)(context.model,batch,context.state).loss
        torch.testing.assert_close(actual_loss,expected_loss)
        expected_loss.backward()
        actual_loss.backward()
        for key,value in context.model.named_parameters():
            torch.testing.assert_close(value.grad,dict(reference.named_parameters())[key].grad,rtol=1e-5,atol=1e-6)
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in context.model.modules())
    finally:
        context.tracker.close()


def test_curriculum_training_evaluation_and_exact_next_updates_resume(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=80)
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,resolve(model_spec('MegaCRN').step)(partial.loss_fn),until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state']
    assert expected['scheduler']==actual['scheduler']
    assert expected['rng']['numpy']==actual['rng']['numpy']
    for key,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)


def test_training_mode_forward_ignores_labels_and_rng_and_teacher_is_explicit(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=80)
    context = make_context(configuration(tmp_path,'inputs'))
    try:
        model = context.model.train()
        batch = next(iter(context.bundle.train))
        inputs = model_inputs(batch,1)
        before = copy.deepcopy(np.random.get_state())
        torch_before = torch.get_rng_state()
        first = model(inputs)
        second = model({**inputs,'y':torch.full_like(batch['y'],float('nan'))})
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        assert before[2:]==after[2:]
        torch.testing.assert_close(torch_before,torch.get_rng_state(),rtol=0,atol=0)
        saved = np.random.get_state()
        a = model.forward_with_aux(inputs,targets=torch.zeros_like(batch['y']),batches_seen=0)[0]
        np.random.set_state(saved)
        b = model.forward_with_aux(inputs,targets=torch.full_like(batch['y'],100.),batches_seen=0)[0]
        assert not torch.allclose(a[:,1:],b[:,1:])
        torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=0)
        assert model.compute_sampling_threshold(10**9)==0.
    finally:
        context.tracker.close()


def test_no_covariate_batch_one_float64_and_cpu_amp(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=80)
    config = configuration(tmp_path,'dtype')
    config['data']['features'] = dict(time_in_day=False)
    context = make_context(config)
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        prediction = model(dict(X=batch['X'][:1].double()))
        assert prediction.shape==(1,3,3,1) and prediction.dtype==torch.float64
        assert torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        assert all(value.grad is None or torch.isfinite(value.grad).all() for value in model.parameters())
    finally:
        context.tracker.close()
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0
