"""Regression checks for data selection, ETA exports and constant scalers."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
from libcity.data.dataset.dataset_subclass.simst_dataset import SimSTDataset
from libcity.data.utils import generate_dataloader
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


def test_simst_reuses_cache_and_keeps_real_evaluation_tail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = SimSTDataset.__new__(SimSTDataset)
    dataset.config = {'data_files': 'fixture', 'output_window': 3, 'y_start': 1}
    dataset.cache_file_name = dataset._get_cache_file_name('point_based')
    dataset.config['y_start'] = 2
    assert dataset._get_cache_file_name('point_based') != dataset.cache_file_name
    dataset.config['y_start'] = 1
    dataset.cache_file_folder = str(Path(dataset.cache_file_name).parent)
    dataset.cache_dataset = dataset.pad_with_last_sample = True
    dataset.data_files = 'fixture'
    dataset.input_window, dataset.output_window, dataset.y_start = 12, 3, 1
    dataset.train_rate, dataset.eval_rate = .5, .25
    dataset.timesolts = np.arange(116)
    dataset.add_time_in_day = dataset.add_day_in_week = False
    dataset._load_dyna = lambda _: np.arange(232, dtype=float).reshape(116, 2, 1)
    dataset._logger = logging.getLogger(__name__)
    dataset._generate_train_val_test()
    monkeypatch.setattr(dataset, '_generate_train_val_test',
                        lambda: pytest.fail('Matching cache should be reused'))
    dataset.output_dim, dataset.num_nodes = 1, 2
    dataset.scaler_type = dataset.ext_scaler_type = 'none'
    dataset.normal_external = False
    dataset.adj_mx, dataset.in_neighbor_num = np.ones((2, 2)), 1
    dataset.batch_size = dataset.infer_bs = 7
    dataset.num_workers = 0
    dataset.feature_name = {'X': 'float', 'y': 'float', 'node_idx': 'int'}

    loaders = dataset.get_data()
    assert [len(loader.dataset) for loader in loaders] == [49, 15, 15]
    assert dataset.get_data() == loaders


def test_legacy_cache_is_preserved_and_not_reused(traffic_data):
    dataset = TrafficStatePointDataset(traffic_data)
    legacy = Path(dataset.cache_file_folder) / ('point_based_' + dataset.parameters_str + '.npz')
    np.savez_compressed(legacy, sentinel=np.array([999.]))
    assert str(legacy) != dataset.cache_file_name
    assert dataset.get_data()[2].dataset[0][0].item() == 9.
    with np.load(legacy, allow_pickle=False) as cached:
        assert cached['sentinel'].item() == 999.




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
            assert torch.isfinite(batch['X']).all()
            assert torch.isfinite(batch['y']).all()
            torch.testing.assert_close(dataset.scaler.inverse_transform(batch['X']),
                                       torch.full_like(batch['X'], data['speed'].iloc[0]))
