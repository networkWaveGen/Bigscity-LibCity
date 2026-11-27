"""Exercise real cache reuse, invalidation and split-independent normalization."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data import traffic_bundle
from libcity.testing.fixtures import atomic_fixture
from runtime_helpers import tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root):
    folder = atomic_fixture(root/'raw', steps=100)
    config = tiny_config(root)
    config['data'].update(cache=True, scaler={'name': 'standard'})
    return folder, load_config(config=config)


def prepare(config):
    return traffic_bundle.prepare_traffic(config['data'], config['training'], config['evaluation'], 17)


def test_cache_hit_avoids_atomic_reader_and_recomputes_windows_and_scaler(tmp_path, monkeypatch):
    _, config = configuration(tmp_path)
    original = prepare(config)
    assert original.split_summary['cache']['status'] == 'miss'
    cache = Path(original.split_summary['cache']['path'])
    assert cache.is_relative_to(tmp_path/'cache')
    before = cache.stat().st_mtime_ns
    with np.load(cache, allow_pickle=False) as saved:
        # Persist raw values, never values normalized using a trial's training split.
        assert saved['target'][6, 1, 0] == 0
        assert saved['target'].shape == (100, 3, 1)
    def forbidden(*args, **kwargs):
        pytest.fail('A cache hit must not parse the atomic source again')
    monkeypatch.setattr(traffic_bundle, '_read_atomic', forbidden)
    cached = prepare(config)
    torch.testing.assert_close(next(iter(original.train))['X'], next(iter(cached.train))['X'])
    config['data'].update(input_window=6, output_window=3, split={'train': .5, 'val': .25},
                          scaler={'name': 'none'})
    config['training']['batch_size'] = 5
    changed = prepare(config)
    assert changed.split_summary['cache']['status'] == 'hit'
    assert next(iter(changed.train))['X'].shape[1] == 6
    assert next(iter(changed.train))['y'].shape[1] == 3
    assert changed.val.dataset.target_starts[0] == 50
    assert cache.stat().st_mtime_ns == before


@pytest.mark.parametrize('source', ['dyna', 'rel', 'geo', 'config'])
def test_source_changes_invalidate_cache(tmp_path, source):
    folder, config = configuration(tmp_path)
    original = prepare(config)
    if source == 'config':
        path = folder/'config.json'
        metadata = json.loads(path.read_text())
        metadata['time_intervals'] = 300
        path.write_text(json.dumps(metadata, indent=2))
    else:
        path = folder/('Tiny.'+source)
        frame = pd.read_csv(path)
        if source == 'dyna':
            frame['speed'] += 100
        elif source == 'rel':
            frame['distance'] += 1
        else:
            frame['coordinates'] = '[1,2]'
        frame.to_csv(path, index=False)
    changed = prepare(config)
    assert changed.split_summary['cache']['status'] == 'miss'
    assert prepare(config).split_summary['cache']['status'] == 'hit'
    if source == 'dyna':
        assert changed.scaler.mean == pytest.approx(original.scaler.mean+100)


def test_feature_graph_and_external_sources_have_independent_identity(tmp_path):
    folder, config = configuration(tmp_path)
    original = prepare(config)
    config['data']['features']['time_in_day'] = True
    calendar = prepare(config)
    assert calendar.metadata.feature_dim == 2
    assert calendar.split_summary['cache']['path'] != original.split_summary['cache']['path']
    times = pd.read_csv(folder/'Tiny.dyna')['time'].drop_duplicates().to_numpy()
    external = folder/'Tiny.ext'
    pd.DataFrame(dict(ext_id=range(100), time=times, holiday=np.arange(100)%2)).to_csv(external, index=False)
    config['data']['features'].update(load_external=True, known_future_columns=['holiday'])
    ext = prepare(config)
    assert prepare(config).split_summary['cache']['status'] == 'hit'
    frame = pd.read_csv(external)
    frame['holiday'] += 1
    frame.to_csv(external, index=False)
    changed = prepare(config)
    assert changed.split_summary['cache']['status'] == 'miss'
    torch.testing.assert_close(next(iter(changed.val))['future_features'][..., -1],
                               next(iter(ext.val))['future_features'][..., -1]+1)
    config['data']['graph']['self_loops'] = True
    graph = prepare(config)
    assert graph.split_summary['cache']['path'] != changed.split_summary['cache']['path']
    np.testing.assert_array_equal(graph.metadata.adjacency.diagonal(), 1)


def test_corrupt_cache_is_rebuilt_and_disabled_cache_does_not_reuse_it(tmp_path):
    _, config = configuration(tmp_path)
    original = prepare(config)
    path = Path(original.split_summary['cache']['path'])
    path.write_bytes(b'interrupted cache')
    rebuilt = prepare(config)
    assert rebuilt.split_summary['cache']['status'] == 'miss'
    torch.testing.assert_close(next(iter(original.train))['y_raw'], next(iter(rebuilt.train))['y_raw'])
    config['data']['cache'] = False
    before = path.stat().st_mtime_ns
    assert prepare(config).split_summary['cache']['status'] == 'disabled'
    assert path.stat().st_mtime_ns == before
