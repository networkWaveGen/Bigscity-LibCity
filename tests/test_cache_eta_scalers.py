"""Regression checks for data selection, ETA exports and constant scalers."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
from libcity.data.utils import generate_dataloader
from libcity.evaluator.eta_evaluator import ETAEvaluator
from libcity.executor.eta_executor import ETAExecutor
from libcity.utils.normalization import NormalScaler, StandardScaler, MinMax01Scaler, MinMax11Scaler


@pytest.fixture
def traffic_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / 'raw_data/fixture'
    raw.mkdir(parents=True)
    pd.DataFrame({'geo_id': [0], 'type': ['Point'], 'coordinates': ['[0,0]']}).to_csv(
        raw / 'fixture.geo', index=False)
    data = pd.DataFrame(dict(dyna_id=range(12), type=['state'] * 12,
        time=pd.date_range('2020-01-01', periods=12, freq='5min').strftime('%Y-%m-%dT%H:%M:%SZ'),
        entity_id=[0] * 12, speed=np.arange(1, 13), flow=np.arange(101, 113)))
    data.to_csv(raw / 'fixture.dyna', index=False)
    other = data.copy()
    other['speed'] += 200
    other.to_csv(raw / 'other.dyna', index=False)
    return dict(dataset='fixture', batch_size=2, input_window=1, output_window=1,
                train_rate=.5, eval_rate=.2, cache_dataset=True, scaler='none',
                num_workers=0, pad_with_last_sample=False, data_col=['speed'])


@pytest.mark.parametrize(('selection', 'expected'), [
    ({'data_col': ['flow']}, 109.), ({'data_files': 'other'}, 209.),
])
def test_cache_separates_selected_columns_and_source_files(traffic_data, selection, expected, monkeypatch):
    original = TrafficStatePointDataset(traffic_data)
    assert original.get_data()[2].dataset[0][0].item() == 9.
    selected = TrafficStatePointDataset({**traffic_data, **selection})
    assert selected.cache_file_name != original.cache_file_name
    assert selected.get_data()[2].dataset[0][0].item() == expected
    repeated = TrafficStatePointDataset({**traffic_data, **selection})
    monkeypatch.setattr(repeated, '_generate_train_val_test',
                        lambda: pytest.fail('Matching cache should be reused'))
    assert repeated.get_data()[2].dataset[0][0].item() == expected
    assert Path(original.cache_file_name).exists()


@pytest.mark.parametrize('metadata', ['missing', 'mismatched'])
def test_cache_rebuilds_missing_or_mismatched_metadata(traffic_data, metadata):
    original = TrafficStatePointDataset(traffic_data)
    original.get_data()
    with np.load(original.cache_file_name, allow_pickle=False) as cached:
        arrays = {name: cached[name] for name in cached.files if name != 'cache_config'}
    for array in arrays.values():
        array.fill(999.)
    if metadata == 'mismatched':
        arrays['cache_config'] = np.asarray('{}')
    np.savez_compressed(original.cache_file_name, **arrays)
    rebuilt = TrafficStatePointDataset(traffic_data)
    assert rebuilt.get_data()[2].dataset[0][0].item() == 9.
    with np.load(rebuilt.cache_file_name, allow_pickle=False) as cached:
        assert cached['cache_config'].item() == rebuilt._cache_config_json()


def test_legacy_cache_is_preserved_and_not_reused(traffic_data):
    dataset = TrafficStatePointDataset(traffic_data)
    legacy = Path(dataset.cache_file_folder) / ('point_based_' + dataset.parameters_str + '.npz')
    np.savez_compressed(legacy, sentinel=np.array([999.]))
    assert str(legacy) != dataset.cache_file_name
    assert dataset.get_data()[2].dataset[0][0].item() == 9.
    with np.load(legacy, allow_pickle=False) as cached:
        assert cached['sentinel'].item() == 999.


@pytest.mark.parametrize('batch_size', [1, 2])
def test_eta_export_keeps_all_trajectories_for_repeated_users(tmp_path, monkeypatch, batch_size):
    monkeypatch.chdir(tmp_path)
    runner = object.__new__(ETAExecutor)
    runner.config = dict(model='DeepTTE', dataset='fixture', metrics=['MAE'], save_modes=['json'])
    runner.device = torch.device('cpu')
    runner.output_pred = True
    runner._logger = logging.getLogger(__name__)
    runner.evaluate_res_dir = str(tmp_path)
    runner.evaluator = ETAEvaluator(runner.config)

    class Predictor(torch.nn.Module):
        def predict(self, batch):
            return batch['time'] + 1.

    runner.model = Predictor()
    samples = [([10.], [1], [101], [116.1], [39.1], [1], [0]),
               ([20.], [1], [102], [116.2], [39.2], [1], [0]),
               ([30.], [2], [201], [116.3], [39.3], [1], [0])]
    fields = dict(time='float', uid='int', traj_id='int', current_longi='float',
                  current_lati='float', traj_len='int', start_timestamp='int')
    _, _, test = generate_dataloader(samples, samples, samples, fields, batch_size, 0)
    runner.evaluate(test)
    result = json.loads(next(tmp_path.glob('*_predictions.json')).read_text())
    assert set(result) == {'1', '2'}
    assert set(result['1']) == {'101', '102'}
    assert set(result['2']) == {'201'}
    assert result['1']['101']['truth'] == 10.
    assert result['1']['102']['prediction'] == 21.
    assert runner.evaluator.evaluate()['MAE'] == pytest.approx(1.)


def scaler_case(kind, constant):
    scale = 0. if constant else 2.
    if kind == 'normal':
        return NormalScaler(scale), 0., 1. if constant else scale, False
    if kind == 'standard':
        return StandardScaler(5., scale), 5., 1. if constant else scale, False
    cls = MinMax01Scaler if kind == 'minmax01' else MinMax11Scaler
    return cls(5., 5. + scale), 5., 1. if constant else scale, kind == 'minmax11'


@pytest.mark.parametrize('kind', ['normal', 'standard', 'minmax01', 'minmax11'])
@pytest.mark.parametrize('constant', [False, True])
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_scaler_finite_roundtrip_and_nonconstant_behavior(kind, constant, backend):
    scaler, offset, scale, symmetric = scaler_case(kind, constant)
    raw = np.array([offset, offset + 2.], dtype=np.float32)
    data = torch.tensor(raw, requires_grad=True) if backend == 'torch' else raw
    transformed = scaler.transform(data)
    expected = (raw - offset) / scale
    if symmetric:
        expected = 2. * expected - 1.
    if backend == 'torch':
        torch.testing.assert_close(transformed, torch.tensor(expected))
        torch.testing.assert_close(scaler.inverse_transform(transformed), data)
        transformed.square().sum().backward()
        assert torch.isfinite(transformed).all() and torch.isfinite(data.grad).all()
    else:
        assert np.isfinite(transformed).all()
        np.testing.assert_allclose(transformed, expected)
        np.testing.assert_allclose(scaler.inverse_transform(transformed), raw)


@pytest.mark.parametrize('kind', ['normal', 'standard', 'minmax01', 'minmax11'])
def test_constant_data_is_finite_through_real_dataset(traffic_data, kind):
    path = Path('raw_data/fixture/fixture.dyna')
    data = pd.read_csv(path)
    data['speed'] = 0. if kind == 'normal' else 5.
    data.to_csv(path, index=False)
    dataset = TrafficStatePointDataset({**traffic_data, 'scaler': kind})
    for loader in dataset.get_data():
        for batch in loader:
            batch.to_tensor('cpu')
            assert torch.isfinite(batch['X']).all()
            assert torch.isfinite(batch['y']).all()
            torch.testing.assert_close(dataset.scaler.inverse_transform(batch['X']),
                                       torch.full_like(batch['X'], data['speed'].iloc[0]))
