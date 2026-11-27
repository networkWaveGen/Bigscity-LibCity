"""Verify migrated algorithms and full lifecycles, not import-only support."""
import importlib

import pytest
import torch

from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.runtime_registry import model_spec, resolve
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context, tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')

PARAMS = {
    'AutoEncoder': {},
    'AGCRN': dict(rnn_units=4, embed_dim=2, num_layers=1, cheb_order=3),
    'TGCN': dict(rnn_units=4, regularization=.02),
    'TrafficRNN': dict(hidden_size=4),
    'TrafficLSTM': dict(hidden_size=4),
    'Seq2Seq': dict(hidden_size=4),
}


def configuration(root, model, experiment):
    config = tiny_config(root, experiment, epochs=2, shuffle=True, batch_size=8)
    config['model'] = dict(name=model, params=dict(PARAMS[model]))
    config['training']['scheduler'] = dict(name='step', interval='epoch', step_size=1, gamma=.8)
    return config


def reference(context, name):
    metadata = context.bundle.metadata
    config = dict(context.config['model']['params'], input_window=4, output_window=2,
                  device=torch.device('cpu'))
    if name == 'TGCN':
        config['lambda'] = config.pop('regularization')
    old_name = 'RNN' if name in ('TrafficRNN', 'TrafficLSTM') else name
    domain = 'traffic_flow_prediction' if name == 'AGCRN' else 'traffic_speed_prediction'
    old = getattr(importlib.import_module(f'libcity.model.{domain}.{old_name}'), old_name)
    return old(config, dict(num_nodes=metadata.num_nodes, feature_dim=metadata.feature_dim,
        output_dim=metadata.output_dim, adj_mx=metadata.adjacency.copy(), scaler=NoneScaler()))


@pytest.mark.parametrize('name', PARAMS)
def test_reference_predictions_and_gradients(tmp_path, name, monkeypatch):
    atomic_fixture(tmp_path/'raw', steps=60)
    context = make_context(configuration(tmp_path, name, 'reference'))
    try:
        old = reference(context, name).eval()
        current = context.model.eval()
        loaded = current.load_state_dict(old.state_dict(), strict=False)
        assert not loaded.unexpected_keys
        assert all(key.endswith('normalized_adj') for key in loaded.missing_keys)
        batch = next(iter(context.bundle.train))
        if name == 'Seq2Seq':
            # The new deterministic start token intentionally replaces random noise.
            monkeypatch.setattr(torch, 'randn', lambda *shape, **kwargs: torch.zeros(*shape, **kwargs))
        expected = old.predict(batch)
        actual = current(model_inputs(batch, 1))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        if name == 'TGCN':
            old_loss = old.calculate_loss(batch)
            new_loss = resolve(model_spec(name).step)(context.loss_fn)(current, batch, context.state).loss
            torch.testing.assert_close(new_loss, old_loss, rtol=1e-5, atol=1e-6)
        else:
            old_loss, new_loss = expected.square().mean(), actual.square().mean()
        old_loss.backward()
        new_loss.backward()
        for key, value in current.named_parameters():
            torch.testing.assert_close(value.grad, dict(old.named_parameters())[key].grad, rtol=1e-5, atol=1e-6)
        assert all(not any(hasattr(module, key) for key in ('_scaler','_logger','device','_device'))
                   for module in current.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name', PARAMS)
def test_train_evaluate_and_exact_resume(tmp_path, name):
    atomic_fixture(tmp_path/'raw', steps=60)
    def config(experiment):
        value = configuration(tmp_path, name, experiment)
        if name in ('TrafficRNN','TrafficLSTM','Seq2Seq'):
            value['model']['params'].update(num_layers=2, dropout=.1, teacher_forcing_ratio=.5)
        return value
    result = run_experiment('train', config=config('full'))
    evaluated = run_experiment('evaluate', config=config('evaluate'), checkpoint=result.artifacts['best'])
    assert evaluated.metrics['loss'] == result.metrics['loss']
    expected = torch.load(result.artifacts['last'], weights_only=True)
    partial = make_context(config('partial'))
    step = model_spec(name).step
    train_loop(partial, resolve(step)(partial.loss_fn) if step else None, until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train', config=config('resume'), resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'], weights_only=True)
    assert expected['train_state'] == actual['train_state']
    assert expected['scheduler'] == actual['scheduler']
    for key, value in expected['model'].items():
        torch.testing.assert_close(value, actual['model'][key], rtol=0, atol=0)


@pytest.mark.parametrize('name', PARAMS)
def test_label_free_batch_one_and_dtype_migration(tmp_path, name):
    atomic_fixture(tmp_path/'raw', steps=60)
    context = make_context(configuration(tmp_path, name, 'dtype'))
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        inputs = {key: value[:1].double() for key, value in model_inputs(batch, 1).items()}
        before = torch.get_rng_state()
        expected = model(inputs)
        actual = model({**inputs, 'y': torch.full((1, 2, 3, 1), float('nan'))})
        assert actual.shape == (1, 2, 3, 1) and actual.dtype == torch.float64
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        if name == 'TGCN':
            assert dict(model.named_buffers())['tgcn_model.normalized_adj'].dtype == torch.float64
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name', ['TrafficRNN','TrafficLSTM','Seq2Seq'])
def test_teacher_forcing_is_explicit_and_changes_only_training_path(tmp_path, name):
    atomic_fixture(tmp_path/'raw', steps=60)
    config = configuration(tmp_path, name, 'teacher')
    config['model']['params']['teacher_forcing_ratio'] = 1.
    context = make_context(config)
    try:
        model = context.model.eval()
        batch = next(iter(context.bundle.train))
        inputs = model_inputs(batch, 1)
        a = model.forward_with_teacher(inputs, torch.zeros_like(batch['y']))
        b = model.forward_with_teacher(inputs, torch.full_like(batch['y'], 100.))
        assert not torch.allclose(a[:, 1], b[:, 1])
        torch.testing.assert_close(a[:, 0], b[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), model({**inputs, 'y': torch.full_like(batch['y'], float('nan'))}))
    finally:
        context.tracker.close()
