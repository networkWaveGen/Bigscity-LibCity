import json
from pathlib import Path
import subprocess
import sys

import pytest

from libcity.config.runtime import load_config
from libcity.data.map_matching_bundle import prepare_map_matching
from libcity.pipeline.experiment import run_experiment
from runtime_fixtures import map_fixture

ROOT=Path(__file__).resolve().parents[1]


def settings(root):
    return dict(data=dict(dataset='RoadTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache')),
                model={'name':'HMMM'},runtime=dict(output_dir=str(root/'runs'),exp_id='map',tensorboard=True))


@pytest.mark.parametrize('with_time',[False,True])
def test_map_bundle_and_direct_evaluation_without_training_objects(tmp_path,monkeypatch,with_time):
    map_fixture(tmp_path/'raw',with_time=with_time)
    config=settings(tmp_path)
    resolved=load_config(config=config)
    assert resolved['runtime']['task']=='map_matching' and resolved['training']=={}
    assert 'input_window' not in resolved['data']
    bundle=prepare_map_matching(resolved['data'],{},resolved['evaluation'],0)
    assert bundle.train is None and bundle.val is None
    assert set(bundle.test['trajectory'][7])=={101,102}
    assert bundle.metadata.info['features']['with_time']==with_time
    import libcity.pipeline.experiment as pipeline
    monkeypatch.setattr(pipeline,'build_context',lambda *args,**kwargs:pytest.fail('Initialized training context'))
    monkeypatch.setattr(pipeline.CheckpointManager,'load_weights',lambda *args:pytest.fail('Loaded checkpoint'))
    result=run_experiment('evaluate',config=config)
    assert result.metrics==pytest.approx({'RMF':0.,'AN':1.,'AL':1.})
    predictions=json.loads(result.artifacts['predictions'].read_text())
    assert set(predictions['7'])=={'101','102'}
    assert any(row[1]==0 for row in predictions['7']['101'])
    run=tmp_path/'runs/map'
    assert not (run/'metrics.jsonl').exists()
    assert not (run/'tensorboard').exists()
    assert not (run/'checkpoints').exists()
    assert (run/'eval/metrics.csv').exists()
    info=json.loads(result.artifacts['runtime_info'].read_text())
    assert info['split']['test']==2
    assert info['capabilities']=={'requires_checkpoint':False,'supports_resume':False}


def test_map_single_metric_and_prediction_switch_remove_stale_export(tmp_path):
    map_fixture(tmp_path/'raw')
    config=settings(tmp_path)
    config['evaluation']={'metrics':['AN']}
    result=run_experiment('evaluate',config=config)
    assert result.metrics=={'AN':1.}
    path=result.artifacts['predictions']
    assert path.exists()
    result=run_experiment('evaluate',config=config,overrides={'evaluation':{'save_predictions':False}})
    assert 'predictions' not in result.artifacts and not path.exists()


def test_map_capabilities_fail_before_data_or_training_initialization(tmp_path,monkeypatch):
    config=settings(tmp_path)
    monkeypatch.setattr('libcity.data.map_matching_bundle.prepare_map_matching',
                        lambda *args,**kwargs:pytest.fail('Prepared data for invalid capability request'))
    with pytest.raises(ValueError,match='direct evaluation only'):
        run_experiment('train',config=config)
    with pytest.raises(ValueError,match='does not support --resume'):
        run_experiment('evaluate',config=config,resume='missing.pt')
    with pytest.raises(ValueError,match='does not use a checkpoint'):
        run_experiment('evaluate',config=config,checkpoint='unused.pt')
    assert not (tmp_path/'runs').exists()


def test_map_cli_matches_python_with_no_checkpoint(tmp_path):
    map_fixture(tmp_path/'raw')
    config=settings(tmp_path)
    path=tmp_path/'map.json'
    path.write_text(json.dumps(config))
    api=run_experiment('evaluate',path)
    result=subprocess.run([sys.executable,'-B',str(ROOT/'run_model.py'),'evaluate','--config',str(path),
                           '--set','runtime.exp_id=cli'],cwd=tmp_path.parent,capture_output=True,text=True,check=True)
    cli=json.loads(result.stdout.splitlines()[-1])
    assert cli['metrics']==pytest.approx(api.metrics)
