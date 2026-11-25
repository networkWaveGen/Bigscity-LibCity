"""Concentrated regression checks for the modern core, without optional stacks."""

import copy
from importlib import import_module
import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.config_parser import ConfigParser
from libcity.data.batch import Batch, BatchPAD
from libcity.data.utils import BatchCollator, generate_dataloader, generate_dataloader_pad
from libcity.evaluator.traffic_state_evaluator import TrafficStateEvaluator
from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.utils.normalization import NoneScaler
from libcity.registry import get_model


@pytest.fixture(scope='module', autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_core_imports_do_not_load_optional_stacks():
    code = '''
import sys
from libcity.pipeline.pipeline import run_model
from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.model.traffic_speed_prediction.RNN import RNN
assert not {'ray', 'dgl', 'torchtext', 'fastdtw', 'gensim'} & set(sys.modules)
'''
    subprocess.run([sys.executable, '-B', '-c', code], check=True)


@pytest.mark.parametrize('padded', [False, True])
def test_dense_batch_values_and_source_ownership(padded):
    samples = [(np.array([[1., 2.]], dtype=np.float64), [3, 4]),
               (np.array([[5., 6.]], dtype=np.float64), [7, 8])]
    original = copy.deepcopy(samples)
    collate = BatchCollator({'X': 'float', 'ids': 'int'}, padded=padded)
    batch = pickle.loads(pickle.dumps(collate))(samples)
    assert isinstance(batch, BatchPAD if padded else Batch)
    batch['X'][0, 0, 0] = 99
    assert samples[0][0][0, 0] == original[0][0][0, 0]
    batch['X'][0, 0, 0] = 1
    batch.to_tensor('cpu')
    torch.testing.assert_close(batch['X'], torch.tensor([[[1., 2.]], [[5., 6.]]]))
    assert batch['ids'].dtype == torch.long
    batch['X'].zero_()
    for actual, expected in zip(samples, original):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
    batch.to_tensor('cpu')  # A second device conversion accepts existing tensors.


def test_padding_and_ragged_fields_do_not_mutate_samples():
    samples = [([1, 2, 3], [4, 5]), ([6], [7])]
    original = copy.deepcopy(samples)
    batch = BatchCollator({'seq': 'int', 'extra': 'no_pad_int'},
                          {'seq': 0}, {'seq': 2})(samples)
    np.testing.assert_array_equal(batch['seq'], [[2, 3], [6, 0]])
    assert batch.get_origin_len('seq') == [2, 1]
    batch.to_tensor('cpu')
    batch['extra'][0][0] = 99
    assert samples == original


def test_padded_loader_keeps_batch_type_without_padding_fields():
    loader, _, _ = generate_dataloader_pad(
        [([1, 2],)], [([1, 2],)], [([1, 2],)], {'ids': 'int'}, 1, 0, None)
    assert isinstance(next(iter(loader)), BatchPAD)


def test_padded_collator_rejects_extra_sample_fields():
    with pytest.raises(KeyError):
        BatchCollator({'ids': 'int'}, {'ids': 0})([([1], 'unexpected')])


def test_loader_works_with_spawn_worker():
    samples = [(np.ones((2, 3, 1)), np.ones((1, 3, 1))) for _ in range(3)]
    loader, _, _ = generate_dataloader(samples, samples, samples,
                                       {'X': 'float', 'y': 'float'}, 2, 1)
    batches = list(loader)
    assert sum(len(batch['X']) for batch in batches) == 3


def test_last_sample_padding_handles_different_input_and_target_shapes():
    samples = [(np.full((4, 2, 1), i), np.full((1, 2, 1), i)) for i in range(3)]
    loader, valid_loader, test_loader = generate_dataloader(samples, samples, samples,
                                       {'X': 'float', 'y': 'float'}, 2, 0,
                                       shuffle=False, pad_with_last_sample=True)
    batches = list(loader)
    batches[-1].to_tensor('cpu')
    assert batches[-1]['X'].shape == (2, 4, 2, 1)
    assert batches[-1]['y'].shape == (2, 1, 2, 1)
    assert batches[-1]['X'][:, 0, 0, 0].tolist() == [2, 2]
    assert sum(len(batch['X']) for batch in valid_loader) == 3
    assert sum(len(batch['X']) for batch in test_loader) == 3


@pytest.mark.parametrize('use_gpu', [False, True])
def test_cpu_device_selection(monkeypatch, use_gpu):
    parser = object.__new__(ConfigParser)
    parser.config = {'gpu': use_gpu}
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'set_device',
                        lambda *_: pytest.fail('CPU path called set_device'))
    parser._init_device()
    assert parser.config['device'] == torch.device('cpu')


def test_config_name_preserves_dotted_stem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'train.v2.json').write_text('{"hidden_size": 17}', encoding='utf-8')
    (tmp_path / 'train.json').write_text('{"hidden_size": 23}', encoding='utf-8')

    parser = object.__new__(ConfigParser)
    parser.config = {}
    parser._parse_config_file('train.v2')

    assert parser['hidden_size'] == 17


def test_packaged_defaults_and_explicit_json_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'raw_data' / 'fixture'
    data.mkdir(parents=True)
    (data / 'config.json').write_text('{}', encoding='utf-8')
    config = tmp_path / 'custom.json'
    config.write_text(json.dumps({'hidden_size': 7}), encoding='utf-8')
    parsed = ConfigParser('traffic_state_pred', 'GRU', 'fixture',
                          config_file=str(config), other_args={'gpu': False})
    assert parsed['hidden_size'] == 7
    assert parsed['model'] == 'RNN'


def test_factory_does_not_hide_constructor_errors(monkeypatch):
    rnn = import_module('libcity.model.traffic_speed_prediction.RNN')

    class Broken:
        def __init__(self, *_):
            raise AttributeError('specific constructor failure')

    monkeypatch.setattr(rnn, 'RNN', Broken)
    with pytest.raises(AttributeError, match='specific constructor failure'):
        get_model({'task': 'traffic_state_pred', 'model': 'RNN'}, {})


@pytest.mark.parametrize('model_name', ['GRU', 'STGCN'])
def test_forecast_forward_backward_and_checkpoint(tmp_path, model_name):
    torch.manual_seed(17)
    config = {'task': 'traffic_state_pred', 'model': model_name,
              'device': torch.device('cpu'), 'input_window': 12, 'output_window': 2,
              'hidden_size': 4, 'num_layers': 1, 'dropout': 0,
              'rnn_type': 'GRU', 'max_epoch': 1, 'Ks': 2, 'Kt': 2,
              'blocks': [[1, 4, 4], [4, 4, 4]]}
    if model_name == 'GRU':
        config['model'] = 'RNN'
    features = {'num_nodes': 3, 'feature_dim': 1, 'output_dim': 1,
                'scaler': NoneScaler(), 'adj_mx': np.ones((3, 3), dtype=np.float32)}
    model = get_model(config, features)
    batch = {'X': torch.rand(2, 12, 3, 1) + 1, 'y': torch.rand(2, 2, 3, 1) + 1}
    prediction = model.predict(batch)
    assert prediction.shape == (2, 2, 3, 1)
    value = model.calculate_loss(batch)
    assert torch.isfinite(value)
    value.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    from libcity.utils.checkpoint import load_checkpoint
    path = tmp_path / 'checkpoint.pt'
    torch.save((model.state_dict(), torch.optim.Adam(model.parameters()).state_dict()), path)
    restored, _ = load_checkpoint(path)
    model.load_state_dict(restored)


@pytest.mark.parametrize('mode', [None, 'refine', 'discover'])
def test_stnn_forecasts_on_cpu(monkeypatch, mode):
    monkeypatch.setattr(torch, 'set_default_tensor_type',
                        lambda *_: pytest.fail('STNN changed the global tensor type'))
    device = torch.device('cpu')
    config = {'task': 'traffic_state_pred', 'model': 'STNN', 'device': device,
              'input_window': 12, 'output_window': 2, 'mode': mode}
    model = get_model(config, {
        'scaler': NoneScaler(), 'num_nodes': 41,
        'adj_mx': np.roll(np.eye(41, dtype=np.float32), 1, axis=1),
    })
    batch = {'X': torch.rand(2, 12, 41, 1, device=device),
             'y': torch.ones(2, 2, 41, 1, device=device)}

    prediction = model.predict(batch)
    assert prediction.shape == batch['y'].shape
    assert prediction.device == model.relations.device == device
    assert torch.isfinite(prediction).all()
    prediction.mean().backward()
    assert model.factors.grad is not None


def test_dsan_forecasts_and_trains_on_cpu():
    device = torch.device('cpu')
    config = {'task': 'traffic_state_pred', 'model': 'DSAN', 'device': device,
              'input_window': 2, 'output_window': 2, 'L': 1,
              'l_d': 1, 'r_d': 0}
    model = get_model(config, {
        'scaler': NoneScaler(), 'len_row': 2, 'len_column': 2, 'num_nodes': 4,
        'feature_dim': 3, 'ext_dim': 1, 'output_dim': 2,
    })
    batch = {'X': torch.rand(2, 2, 2, 2, 3, device=device),
             'y': torch.ones(2, 2, 2, 2, 2, device=device)}

    prediction = model.predict(batch)
    assert prediction.shape == batch['y'].shape
    assert prediction.device == device
    loss = model.calculate_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.dsan.final_layer.weight.grad is not None


@pytest.mark.parametrize('mode', ['single', 'average'])
@pytest.mark.parametrize('save_predictions', [False, True])
def test_evaluation_preserves_global_metrics_and_prediction_archive(
        tmp_path, monkeypatch, mode, save_predictions):
    monkeypatch.chdir(tmp_path)
    truth = torch.tensor([0., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]).reshape(3, 2, 2, 1)
    prediction = truth + torch.tensor([1., -1., 2.]).reshape(3, 1, 1, 1)
    config = {'evaluator': 'TrafficStateEvaluator', 'evaluator_mode': mode,
              'metrics': ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE', 'masked_MSE',
                          'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR'],
              'save_mode': ['csv'], 'device': torch.device('cpu'), 'exp_id': 101,
              'model': 'fixture', 'dataset': 'fixture', 'save_predictions': save_predictions}

    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def predict(self, batch):
            return batch['X'] + self.bias

        def calculate_loss(self, batch):
            return (self.predict(batch) - batch['y']).square().mean()

    reference = TrafficStateEvaluator(config)
    reference.collect({'y_true': truth.clone(), 'y_pred': prediction.clone()})
    expected = reference.evaluate()
    samples = [(x.numpy(), y.numpy()) for x, y in zip(prediction, truth)]
    _, _, loader = generate_dataloader(samples, samples, samples,
                                       {'X': 'float', 'y': 'float'}, 2, 0)
    executor = TrafficStateExecutor(config, Identity(), {'scaler': NoneScaler()})
    try:
        actual = executor.evaluate(loader)
    finally:
        executor._writer.close()
    for key, value in expected.items():
        name, step = key.rsplit('@', 1)
        assert actual[name].iloc[int(step) - 1] == pytest.approx(value, nan_ok=True)
    files = list((tmp_path / 'libcity/cache/101/evaluate_cache').glob('*.npz'))
    assert bool(files) == save_predictions
    if files:
        with np.load(files[0]) as data:
            np.testing.assert_array_equal(data['prediction'], prediction.numpy())
            np.testing.assert_array_equal(data['truth'], truth.numpy())


def test_atomic_files_to_training_and_evaluation(tmp_path, monkeypatch):
    """Exercise the real data loader, config defaults, runner and saved checkpoint."""
    from datetime import datetime, timedelta
    from libcity.pipeline.pipeline import run_model

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / 'raw_data' / 'Tiny'
    folder.mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps({
        'info': {'data_col': ['speed'], 'output_dim': 1, 'time_intervals': 300}}),
        encoding='utf-8')
    pd.DataFrame({'geo_id': [0, 1, 2], 'type': ['Point'] * 3,
                  'coordinates': ['[0,0]', '[1,0]', '[2,0]']}).to_csv(folder / 'Tiny.geo', index=False)
    pd.DataFrame({'rel_id': [0, 1, 2], 'type': ['geo'] * 3,
                  'origin_id': [0, 1, 2], 'destination_id': [1, 2, 0]}).to_csv(
                      folder / 'Tiny.rel', index=False)
    start = datetime(2024, 1, 1)
    rows = [{'dyna_id': node * 48 + t, 'type': 'state',
             'time': (start + timedelta(minutes=t * 5)).strftime('%Y-%m-%dT%H:%M:%SZ'),
             'entity_id': node, 'speed': 10 + node + np.sin(t / 4)}
            for node in range(3) for t in range(48)]
    pd.DataFrame(rows).to_csv(folder / 'Tiny.dyna', index=False)
    run_model('traffic_state_pred', 'GRU', 'Tiny', other_args={
        'gpu': False, 'exp_id': 202, 'max_epoch': 1, 'batch_size': 8,
        'num_workers': 0, 'input_window': 4, 'output_window': 2,
        'hidden_size': 4, 'num_layers': 1, 'dropout': 0,
        'train_rate': 0.6, 'eval_rate': 0.2, 'cache_dataset': False,
        'load_external': False, 'add_time_in_day': False, 'add_day_in_week': False,
        'log_level': 'WARNING', 'scaler': 'standard'})
    directory = tmp_path / 'libcity/cache/202'
    assert (directory / 'model_cache/GRU_Tiny.m').is_file()
    result_file = next((directory / 'evaluate_cache').glob('*.csv'))
    frame = pd.read_csv(result_file)
    assert len(frame) == 2
    assert np.isfinite(frame[['MAE', 'RMSE']].to_numpy()).all()
