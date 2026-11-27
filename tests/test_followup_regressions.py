"""Exercise the runtime paths missed by the first regression batch."""

import copy
import importlib
import json
import logging
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libcity.config.config_parser import ConfigParser
from libcity.data.dataset.traffic_state_datatset import TrafficStateDataset
from libcity.data.utils import generate_dataloader
from libcity.model.traffic_od_prediction.CSTN import CSTN
from libcity.utils.normalization import NoneScaler


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'raw_data/fixture'
    data.mkdir(parents=True)
    (data / 'config.json').write_text('{}', encoding='utf-8')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('count', [1, 2, 3])
def test_cstn_actual_batch_size_and_gradient(count):
    model = CSTN(dict(device=torch.device('cpu'), batch_size=2, input_window=1,
                      output_window=1, output_dim=1, n_layers=1, c_lt=2),
                 dict(scaler=NoneScaler(), len_row=1, len_column=1))
    batch = dict(X=torch.ones(count, 1, 1, 1, 1, 1, 1), W=torch.ones(count, 1, 29),
                 y=torch.ones(count, 1, 1, 1, 1, 1, 1))
    output = model.predict(batch)
    assert output.shape == batch['y'].shape
    loss = model.calculate_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


class MeanLoss(torch.nn.Module):
    def forward(self, batch, *_):
        return torch.zeros_like(batch['y'])

    def predict(self, batch):
        return self(batch)

    def calculate_loss(self, batch, *_):
        return batch['y'].abs().mean()

    calculate_val_loss = calculate_loss

    def calculate_loss_without_predict(self, truth, prediction, **kwargs):
        return (truth - prediction).abs().mean()


@pytest.mark.parametrize(('module', 'name'), [
    ('traffic_state_executor', 'TrafficStateExecutor'), ('dcrnn_executor', 'DCRNNExecutor'),
    ('mtgnn_executor', 'MTGNNExecutor'), ('pdformer_executor', 'PDFormerExecutor'),
    ('astgnn_executor', 'ASTGNNExecutor'), ('sstban_executor', 'SSTBANExecutor'),
    ('fogs_executor', 'FOGSExecutor'),
])
@pytest.mark.parametrize('empty', [False, True])
def test_all_batched_validation_overrides_account_for_real_samples(module, name, empty):
    cls = getattr(importlib.import_module('libcity.executor.' + module), name)
    runner = object.__new__(cls)
    runner.model = MeanLoss()
    runner.device = torch.device('cpu')
    runner._logger = logging.getLogger(__name__)
    runner._writer = SimpleNamespace(add_scalar=lambda *_: None)
    runner.lap_mx = torch.ones(1, 1)
    runner.set_loss = 'mae'
    runner.adjust_output = lambda output, *_: output[..., 0]
    samples = [(np.zeros((1, 1, 1)), np.full((1, 1, 1), value), [0])
               for value in (1., 2., 100.)]
    _, valid, _ = generate_dataloader(samples, [] if empty else samples, samples,
        {'X': 'float', 'y': 'float', 'y_slot': 'int'}, 2, 0, pad_with_last_sample=True)
    if empty:
        with pytest.raises(ValueError, match='at least one sample'):
            runner._valid_epoch(valid, 0)
    else:
        assert runner._valid_epoch(valid, 0) == pytest.approx(103 / 3)


@pytest.mark.parametrize(('json_flags', 'explicit', 'extra', 'expected'), [
    ({'train': False, 'saved_model': False}, {}, {}, (False, False)),
    ({'train': False, 'saved_model': False}, {'train': True, 'saved_model': True}, {}, (True, True)),
    ({'train': True, 'saved_model': True}, {}, {'train': False, 'saved_model': False}, (False, False)),
    ({}, {}, {}, (True, True)),
    ({'train': False, 'saved_model': False}, {}, {'train': None, 'saved_model': None}, (False, False)),
])
def test_real_config_json_flags_and_explicit_overrides(tmp_path, json_flags, explicit, extra, expected):
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(json_flags), encoding='utf-8')
    config = ConfigParser('traffic_state_pred', 'GRU', 'fixture', config_file=str(path),
                          other_args={'gpu': False, **extra}, **explicit)
    assert (config['train'], config['saved_model']) == expected


@pytest.mark.parametrize('arguments', [[], ['--set', 'training.epochs=2']])
def test_real_cli_passes_layered_config_and_explicit_overrides(tmp_path, monkeypatch, arguments):
    path = tmp_path / 'experiment.json'
    path.write_text('{"data": {"dataset": "fixture"}, "model": {"name": "GRU"}, "training": {"epochs": 1}}', encoding='utf-8')
    pipeline = importlib.import_module('libcity.pipeline.experiment')
    observed = []

    def inspect_config(mode, config_file, *, overrides, checkpoint, resume):
        from libcity.config.runtime import load_config
        cfg = load_config(config_file, overrides)
        observed.append((mode, cfg['training']['epochs'], checkpoint, resume))
        return SimpleNamespace(metrics={}, artifacts={})

    monkeypatch.setattr(pipeline, 'run_experiment', inspect_config)
    monkeypatch.setattr(sys, 'argv', ['run_model.py', 'train', '--config', str(path), *arguments])
    runpy.run_path(str(ROOT / 'run_model.py'), run_name='__main__')
    assert observed == [('train', 2 if arguments else 1, None, None)]


def test_json_evaluation_mode_does_not_start_data_preparation(tmp_path, monkeypatch):
    pipeline = importlib.import_module('libcity.pipeline.pipeline')
    path = tmp_path / 'experiment.json'
    path.write_text('{"train": false, "saved_model": false}', encoding='utf-8')
    monkeypatch.setattr(pipeline, 'get_dataset', lambda *_: pytest.fail('Evaluation-only mode started data loading'))
    with pytest.raises(FileNotFoundError, match='Evaluation-only mode'):
        pipeline.run_model('traffic_state_pred', 'GRU', 'fixture', str(path), other_args={'gpu': False})


def test_search_objective_rejects_disabled_training(tmp_path):
    from libcity.pipeline.pipeline import objective_function
    path = tmp_path / 'experiment.json'
    path.write_text('{"train": false}', encoding='utf-8')
    with pytest.raises(ValueError, match='trials require train=true'):
        objective_function('traffic_state_pred', 'GRU', 'fixture', str(path), other_args={'gpu': False})


@pytest.mark.parametrize('cache', [False, True])
def test_real_dataset_get_data_twice_preserves_loaders_values_and_scaler(cache, tmp_path):
    runner = object.__new__(TrafficStateDataset)
    runner.data = None
    runner.cache_dataset = cache
    runner.cache_file_name = str(tmp_path / 'cached.npz')
    arrays = [np.arange(size, dtype=float).reshape(size, 1, 1, 1) + 1 for size in (6, 6, 2, 2, 2, 2)]
    if cache:
        np.savez(runner.cache_file_name, cache_config=np.asarray(runner._cache_config_json()), **dict(zip(
            ('x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test'), arrays)))
    calls = []

    def generate():
        calls.append(True)
        return tuple(value.copy() for value in arrays)

    runner._generate_train_val_test = generate
    runner.robustness_test = False
    runner.scaler_type, runner.ext_scaler_type = 'standard', 'none'
    runner.output_dim, runner.normal_external = 1, False
    runner.feature_name = {'X': 'float', 'y': 'float'}
    runner.batch_size, runner.num_workers, runner.pad_with_last_sample = 2, 0, False
    runner._logger = logging.getLogger(__name__)
    first = runner.get_data()
    mean, std = runner.scaler.mean, runner.scaler.std
    before = [copy.deepcopy(loader.dataset[0]) for loader in first]
    second = runner.get_data()
    assert all(a is b for a, b in zip(first, second))
    assert (runner.scaler.mean, runner.scaler.std) == (mean, std)
    for loader, original in zip(second, before):
        np.testing.assert_array_equal(loader.dataset[0][0], original[0])
    assert len(calls) == (0 if cache else 1)


class Regression(torch.nn.Linear):
    def calculate_loss(self, batch):
        return (self(batch['X'].reshape(-1, 1)) - batch['y'].reshape(-1, 1)).square().mean()


@pytest.mark.parametrize('restore_best', [False, True])
def test_trajectory_best_model_optimizer_scheduler_and_export_agree(tmp_path, restore_best):
    from libcity.executor.traj_loc_pred_executor import TrajLocPredExecutor
    from libcity.utils.checkpoint import load_checkpoint
    runner = object.__new__(TrajLocPredExecutor)
    runner.config = dict(max_epoch=2, learning_rate=.1, clip=100., hyper_tune=False,
                         load_best_epoch=restore_best, early_stop_lr=0., debug=False, device=torch.device('cpu'))
    runner._logger = logging.getLogger(__name__)
    runner.model = Regression(1, 1, bias=False)
    with torch.no_grad():
        runner.model.weight.fill_(.5)
    runner.optimizer = torch.optim.Adam(runner.model.parameters(), lr=.1)
    runner.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(runner.optimizer, mode='max')
    runner.loss_func = None
    runner.cache_dir = str(tmp_path / 'model_cache')
    records = []

    def validation(*_):
        records.append((copy.deepcopy(runner.model.state_dict()), copy.deepcopy(runner.optimizer.state_dict())))
        return ((.9, .1), (.1, .2))[len(records) - 1]

    runner._valid_epoch = validation
    samples = [(np.ones((1,)), np.ones((1,))) for _ in range(2)]
    train, valid, _ = generate_dataloader(samples, samples, samples, {'X': 'float', 'y': 'float'}, 2, 0)
    runner.train(train, valid)
    selected = records[0 if restore_best else 1]
    torch.testing.assert_close(runner.model.weight, selected[0]['weight'])
    assert next(iter(runner.optimizer.state.values()))['step'].item() == (1 if restore_best else 2)
    assert runner.scheduler.last_epoch == (1 if restore_best else 2)
    assert runner.scheduler.num_bad_epochs == (0 if restore_best else 1)
    assert not (tmp_path / 'libcity/tmp/checkpoint').exists()
    path = tmp_path / 'model_cache/export.m'
    runner.save_model(str(path))
    model_state, optimizer_state = load_checkpoint(path)
    torch.testing.assert_close(model_state['weight'], selected[0]['weight'])
    assert next(iter(optimizer_state['state'].values()))['step'].item() == (1 if restore_best else 2)
    # Resume a real next update from the exported pair, not just inspect fields.
    reference = Regression(1, 1, bias=False)
    reference.load_state_dict(selected[0])
    optimizer = torch.optim.Adam(reference.parameters(), lr=.1)
    optimizer.load_state_dict(selected[1])
    batch = next(iter(train))
    batch.to_tensor('cpu')
    expected = reference.calculate_loss(batch)
    optimizer.zero_grad()
    expected.backward()
    optimizer.step()
    runner.run(train, runner.model, .1, 100.)
    torch.testing.assert_close(runner.model.weight, reference.weight)


@pytest.mark.parametrize('stage', ['validation', 'evaluation'])
def test_trajectory_sample_mode_tail_has_matching_targets(stage, tmp_path):
    from libcity.executor.traj_loc_pred_executor import TrajLocPredExecutor
    runner = object.__new__(TrajLocPredExecutor)
    runner.config = dict(device=torch.device('cpu'), evaluate_method='sample', batch_size=2)
    runner.model = MeanLoss()
    runner.metrics = 'Recall@1'
    runner.loss_func = None
    runner.evaluate_res_dir = str(tmp_path)
    counts = []

    def collect(result):
        count = len(result['uid'])
        assert result['loc_true'] == [0] * count
        assert len(result['loc_pred']) == count
        counts.append(count)

    runner.evaluator = SimpleNamespace(clear=counts.clear, collect=collect,
        evaluate=lambda: {'Recall@1': 1.}, save_result=lambda *_: {'Recall@1': 1.})
    samples = [(np.zeros((1,)), np.array([value]), i) for i, value in enumerate((1., 2., 100.))]
    _, valid, _ = generate_dataloader(samples, samples, samples,
        {'X': 'float', 'y': 'float', 'uid': 'int'}, 2, 0)
    if stage == 'validation':
        accuracy, loss = runner._valid_epoch(valid, runner.model)
        assert accuracy == 1.
        assert loss == pytest.approx(103 / 3)
    else:
        runner.evaluate(valid)
    assert counts == [2, 1]
