"""Direct algorithms, isolated trajectory state and unmatched observations."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from libcity.config.runtime import load_config
from libcity.data.map_matching_bundle import prepare_map_matching
from libcity.model.runtime_registry import model_spec, resolve
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import map_fixture

ROOT = Path(__file__).resolve().parents[1]


def configuration(root, name):
    return dict(data=dict(dataset='RoadTiny', data_dir=str(root/'raw')),
        model=dict(name=name), runtime=dict(output_dir=str(root/'runs'), exp_id=name, tensorboard=True))


@pytest.mark.parametrize('name', ['STMatching','IVMM'])
@pytest.mark.parametrize('with_time', [False,True])
def test_multiple_trajectories_and_reference_algorithm(tmp_path, name, with_time, monkeypatch):
    map_fixture(tmp_path/'raw', with_time=with_time)
    config = configuration(tmp_path, name)
    resolved = load_config(config=config)
    bundle = prepare_map_matching(resolved['data'], {}, resolved['evaluation'], 17)
    native = resolve(model_spec(name).factory)(metadata=bundle.metadata, **resolved['model']['params'])
    actual = native.run(bundle.test)
    algorithm_type = getattr(importlib.import_module('libcity.model.map_matching.'+name), name)
    for identity, observations in bundle.test['trajectory'][7].items():
        reference = algorithm_type(resolved['model']['params'], dict(bundle.metadata.info['features']))
        expected = reference.run({**bundle.test, 'trajectory': {7: {identity: observations}}})
        np.testing.assert_array_equal(actual[7][identity], expected[7][identity])
    # Re-running the same object cannot retain an earlier trajectory's state.
    repeated = native.run(bundle.test)
    for identity in actual[7]:
        np.testing.assert_array_equal(actual[7][identity], repeated[7][identity])
    monkeypatch.setattr('libcity.pipeline.experiment.build_context',
        lambda *args, **kwargs: pytest.fail('Direct matching initialized a training context'))
    result = run_experiment('evaluate', config=config)
    assert result.metrics == pytest.approx({'RMF': 0., 'AN': 1., 'AL': 1.})
    assert not (tmp_path/'runs'/name/'checkpoints').exists()
    assert not (tmp_path/'runs'/name/'tensorboard').exists()


@pytest.mark.parametrize('name', ['STMatching','IVMM'])
@pytest.mark.parametrize('case', ['unmatched_tail','all_unmatched','single_point'])
def test_unmatched_and_single_point_trajectories(tmp_path, name, case):
    folder = map_fixture(tmp_path/'raw')
    path = folder/'RoadTiny.dyna'
    frame = pd.read_csv(path)
    if case == 'single_point':
        frame = frame.groupby('traj_id', sort=False).head(1)
    else:
        selected = frame.index if case == 'all_unmatched' else frame.groupby('traj_id', sort=False).tail(1).index
        frame.loc[selected, 'coordinates'] = '[120,30]'
    frame.to_csv(path, index=False)
    result = run_experiment('evaluate', config=configuration(tmp_path, name))
    assert all(np.isfinite(value) for value in result.metrics.values())
    predictions = json.loads(result.artifacts['predictions'].read_text())
    for values in predictions['7'].values():
        assert len(values) == (1 if case == 'single_point' else 6)
        if case == 'all_unmatched':
            assert all(row[1] is None for row in values)
        elif case == 'unmatched_tail':
            assert values[-1][1] is None


@pytest.mark.parametrize('name', ['STMatching','IVMM'])
def test_matching_cli_and_capability_guards(tmp_path, name):
    map_fixture(tmp_path/'raw')
    config = configuration(tmp_path, name)
    for mode, options, message in [('train', {}, 'direct evaluation only'),
        ('evaluate', {'resume': 'missing.pt'}, 'does not support --resume'),
        ('evaluate', {'checkpoint': 'missing.pt'}, 'does not use a checkpoint')]:
        with pytest.raises(ValueError, match=message):
            run_experiment(mode, config=config, **options)
    assert not (tmp_path/'runs').exists()
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    completed = subprocess.run([sys.executable, '-B', str(ROOT/'run_model.py'), 'evaluate', '--config', str(path)],
        cwd=tmp_path.parent, text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout.splitlines()[-1])['metrics'] == pytest.approx({'RMF':0., 'AN':1., 'AL':1.})
