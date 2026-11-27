"""Native calendar, spectral and normalization models retain algorithm semantics."""
import importlib
import json

import pandas as pd
import pytest
import torch

from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context, tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')

PARAMS = {
    'STID': dict(num_block=1, time_series_emb_dim=4, spatial_emb_dim=4, temp_dim_tid=4, temp_dim_diw=4),
    'STAEformer': dict(input_embedding_dim=4, tod_embedding_dim=2, dow_embedding_dim=2,
        spatial_embedding_dim=2, adaptive_embedding_dim=2, feed_forward_dim=16, num_heads=2, num_layers=1, dropout=.1),
    'FreTS': dict(embed_size=4, hidden_size=8),
    'STNorm': dict(blocks=1, layers=2, hidden_channels=4),
    'ATDM': dict(hidden_size=4, past_t=2, dropout=.1),
}


def configuration(root, name, experiment):
    config = tiny_config(root, experiment, epochs=2, batch_size=8, shuffle=True)
    config['model'] = dict(name=name, params=dict(PARAMS[name]))
    config['training']['scheduler'] = dict(name='step', interval='epoch', step_size=1, gamma=.8)
    return config


def multivariate(root):
    folder = atomic_fixture(root/'raw', steps=60)
    frame = pd.read_csv(folder/'Tiny.dyna')
    frame['flow'] = frame['speed']*.5+3
    frame.to_csv(folder/'Tiny.dyna', index=False)
    metadata = json.loads((folder/'config.json').read_text())
    metadata.update(data_col=['speed','flow'], output_dim=2)
    (folder/'config.json').write_text(json.dumps(metadata))


@pytest.mark.parametrize('name', PARAMS)
def test_reference_forward_and_gradients(tmp_path, name):
    atomic_fixture(tmp_path/'raw', steps=60)
    context = make_context(configuration(tmp_path, name, 'reference'))
    try:
        current = context.model.eval()
        metadata = context.bundle.metadata
        config = dict(context.config['model']['params'], input_window=4, output_window=2,
            device=torch.device('cpu'), time_intervals=300, output_dim=1)
        if name == 'STAEformer':
            config.update(input_dim=current.input_dim, steps_per_day=current.steps_per_day,
                          add_time_in_day=True, add_day_in_week=True)
        if name == 'STNorm':
            config['n_pred'] = 2
        domain = 'traffic_flow_prediction' if name in ('FreTS','STNorm') else 'traffic_speed_prediction'
        module = importlib.import_module(f'libcity.model.{domain}.{name}')
        old = getattr(module, name)(config, dict(num_nodes=metadata.num_nodes, feature_dim=metadata.feature_dim,
            output_dim=1, adj_mx=metadata.adjacency.copy(), scaler=NoneScaler())).eval()
        current.load_state_dict(old.state_dict())
        batch = next(iter(context.bundle.train))
        reference_batch = {**batch, 'X': current.packed_inputs(batch['X'])} if name == 'STAEformer' else batch
        expected = old.predict(reference_batch)
        actual = current(model_inputs(batch, 1))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for key, parameter in current.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(old.named_parameters())[key].grad, rtol=1e-5, atol=1e-6)
        assert all(not any(hasattr(module, key) for key in ('_scaler','_logger','device','_device'))
                   for module in current.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name', PARAMS)
def test_complete_training_evaluation_and_exact_resume(tmp_path, name):
    atomic_fixture(tmp_path/'raw', steps=60)
    result = run_experiment('train', config=configuration(tmp_path, name, 'full'))
    evaluated = run_experiment('evaluate', config=configuration(tmp_path, name, 'evaluate'), checkpoint=result.artifacts['best'])
    assert result.metrics['loss'] == evaluated.metrics['loss']
    expected = torch.load(result.artifacts['last'], weights_only=True)
    partial = make_context(configuration(tmp_path, name, 'partial'))
    train_loop(partial, until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train', config=configuration(tmp_path, name, 'resume'), resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'], weights_only=True)
    assert expected['train_state'] == actual['train_state']
    assert expected['scheduler'] == actual['scheduler']
    for key, value in expected['model'].items():
        torch.testing.assert_close(value, actual['model'][key], rtol=0, atol=0)


@pytest.mark.parametrize('name', PARAMS)
def test_batch_one_multivariate_output_and_dtype(tmp_path, name):
    multivariate(tmp_path)
    context = make_context(configuration(tmp_path, name, 'multivariate'))
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        inputs = {key: value[:1].double() for key, value in model_inputs(batch, 2).items()}
        before = torch.get_rng_state()
        prediction = model(inputs)
        assert prediction.shape == (1, 2, 3, 2) and prediction.dtype == torch.float64
        assert torch.isfinite(prediction).all()
        torch.testing.assert_close(prediction, model({**inputs, 'y': torch.full_like(prediction, float('nan'))}))
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        prediction.square().mean().backward()
        assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name', PARAMS)
def test_cpu_amp_training(tmp_path, name):
    atomic_fixture(tmp_path/'raw', steps=60)
    config = configuration(tmp_path, name, 'amp')
    config['training'].update(epochs=1, amp=True, amp_dtype='bfloat16')
    result = run_experiment('train', config=config)
    assert result.metrics['loss'] >= 0


@pytest.mark.parametrize('mixed', [False, True])
def test_staeformer_projection_modes_use_metadata_calendar(tmp_path, mixed):
    multivariate(tmp_path)
    config = configuration(tmp_path, 'STAEformer', 'projection')
    config['model']['params']['use_mixed_proj'] = mixed
    context = make_context(config)
    try:
        model = context.model.eval()
        batch = next(iter(context.bundle.train))
        layout = context.bundle.metadata.input_layout['calendar']
        assert layout['time_in_day'] == 2 and layout['day_of_week'] == tuple(range(3,10))
        packed = model.packed_inputs(batch['X'])
        assert packed.shape[-1] == 4
        torch.testing.assert_close(packed[..., -1], batch['X'][..., 3:10].argmax(dim=-1).float())
        assert model(model_inputs(batch, 2)).shape[-1] == 2
    finally:
        context.tracker.close()


@pytest.mark.parametrize('time,week', [(False,True),(True,False),(False,False)])
def test_stid_independent_calendar_switches_and_unequal_embeddings(tmp_path, time, week):
    atomic_fixture(tmp_path/'raw', steps=60)
    config = configuration(tmp_path, 'STID', 'calendar')
    config['model']['params'].update(if_TiD=time, if_DiW=week, temp_dim_tid=3 if time else 0,
                                    temp_dim_diw=5 if week else 0)
    context = make_context(config)
    try:
        context.model.eval()
        prediction = context.model(model_inputs(next(iter(context.bundle.train)), 1))
        assert prediction.shape[1:] == (2, 3, 1) and torch.isfinite(prediction).all()
    finally:
        context.tracker.close()


def test_stnorm_single_node_statistics_are_finite_and_checkpointed(tmp_path):
    atomic_fixture(tmp_path/'raw', steps=60, nodes=1)
    result = run_experiment('train', config=configuration(tmp_path, 'STNorm', 'single'))
    checkpoint = torch.load(result.artifacts['last'], weights_only=True)
    assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
    assert any(key.endswith('running_mean') for key in checkpoint['model'])
