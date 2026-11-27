import json
from pathlib import Path

import numpy as np
import pytest
import torch

from libcity.config.runtime import load_config, merge_config
from libcity.data.bundle import prepare_traffic
from libcity.model.traffic_speed_prediction.runtime_rnn import GRU
from runtime_fixtures import atomic_fixture


def minimal(root):
    return dict(model={'name': 'GRU'}, data={'dataset': 'Tiny', 'data_dir': str(root / 'raw'),
                'cache_dir': str(root / 'cache')}, runtime={'output_dir': str(root / 'runs')})


def test_config_layers_paths_and_component_replacement(tmp_path, monkeypatch):
    atomic_fixture(tmp_path / 'raw')
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(dict(model={'name': 'GRU', 'params': {'hidden_size': 7}},
        data={'dataset': 'Tiny', 'data_dir': 'raw', 'cache_dir': 'cache'}, runtime={'output_dir': 'runs'},
        training={'optimizer': {'name': 'sgd', 'momentum': .9}})))
    monkeypatch.chdir(tmp_path.parent)
    cfg = load_config(path, {'training': {'batch_size': 3}})
    assert cfg['training']['optimizer'] == {'name': 'sgd', 'momentum': .9}
    assert cfg['training']['batch_size'] == 3 and cfg['model']['params']['hidden_size'] == 7
    assert cfg['data']['data_dir'] == str(tmp_path / 'raw')
    assert cfg['runtime']['output_dir'] == str(tmp_path / 'runs')
    low = {'training': {'optimizer': {'name': 'adam', 'lr': .1}}}
    assert merge_config(low, {'training': {'optimizer': {'lr': .2}}})['training']['optimizer']['lr'] == .2
    assert low['training']['optimizer']['lr'] == .1


@pytest.mark.parametrize('changes,match', [
    ({'model': {}}, 'model.name'), ({'model': {'name': 'Missing'}}, 'not migrated'),
    ({'extra': True}, 'Unknown'), ({'training': {'amp': 'false'}}, 'boolean'),
    ({'training': {'epochs': 0}}, 'integer'), ({'evaluation': {'metrics': ['typo']}}, 'metric'),
    ({'data': {'split': {'train': .9, 'val': .2}}}, 'split'),
    ({'training': {'scheduler': {'name': 'plateau', 'interval': 'step'}}}, 'metric'),
])
def test_invalid_config_fails_at_entry(tmp_path, changes, match):
    base = minimal(tmp_path)
    if changes == {'model': {}}:
        base['model'] = {}
    else:
        base = merge_config(base, changes)
    with pytest.raises((ValueError, TypeError), match=match):
        load_config(config=base)


def test_adapter_matches_atomic_baseline_and_metadata_is_readonly(tmp_path, monkeypatch):
    atomic_fixture(tmp_path / 'raw')
    # Data paths are explicit; no cwd mutation is required by the adapter.
    monkeypatch.chdir(tmp_path.parent)
    cfg = minimal(tmp_path)
    cfg['data'].update(input_window=4, output_window=2, scaler={'name': 'none'}, pad_training=False)
    cfg['training'] = dict(batch_size=2, shuffle=False)
    with pytest.warns(UserWarning, match='protocol'):
        cfg = load_config(config=cfg)
    bundle = prepare_traffic(cfg['data'], cfg['training'], cfg['evaluation'], 17)
    golden = json.loads((Path(__file__).resolve().parents[1] / 'docs/modernization/runtime/baseline/gru.json').read_text())
    batch = next(iter(bundle.train))
    torch.testing.assert_close(batch['X'], torch.tensor(golden['X']))
    torch.testing.assert_close(batch['y'], torch.tensor(golden['y']))
    assert type(batch) is dict and all(isinstance(item, torch.Tensor) for item in batch.values())
    assert sum(len(item['X']) for item in bundle.test) == len(bundle.test.dataset)
    assert bundle.metadata.num_nodes == 3
    with pytest.raises(TypeError):
        bundle.metadata.info['output_dim'] = 9
    with pytest.raises(ValueError):
        bundle.metadata.adjacency[0, 0] = 3
    assert (tmp_path / 'cache').is_dir()
    assert not (tmp_path.parent / 'libcity/cache').exists()
    model = GRU(metadata=bundle.metadata, input_window=4, output_window=2, hidden_size=4)
    model.load_state_dict({name: torch.tensor(value) for name, value in golden['weights'].items()})
    model.eval()
    prediction = model({'X': batch['X']})
    torch.testing.assert_close(prediction, torch.tensor(golden['prediction']), rtol=1e-6, atol=1e-6)
    mask = batch['y'] != 0
    value = (prediction - batch['y']).abs()[mask].mean()
    assert value.item() == pytest.approx(golden['masked_mae'], rel=1e-6)
    value.backward()
    assert model.fc.weight.grad is not None
    assert not any(hasattr(model, key) for key in ('_scaler', 'device', '_logger'))


def test_calendar_feature_does_not_implicitly_enable_external_file(tmp_path):
    folder = atomic_fixture(tmp_path/'raw')
    (folder/'Tiny.ext').write_text('invalid external file must not be read', encoding='utf-8')
    cfg = minimal(tmp_path)
    cfg['data']['features'] = {'time_in_day':True,'load_external':False}
    cfg = load_config(config=cfg)
    bundle = prepare_traffic(cfg['data'],cfg['training'],cfg['evaluation'],17)
    assert bundle.metadata.feature_dim==2
