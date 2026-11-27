import json
from pathlib import Path
import subprocess
import sys
import pytest

from libcity.pipeline.runtime_tuning import run_search,candidates
from libcity.testing.fixtures import atomic_fixture,task_fixture
from runtime_helpers import tiny_config
from test_runtime_tasks import configuration


def test_trials_never_iterate_test_and_reprepare_changed_data(tmp_path,monkeypatch):
    import libcity.pipeline.experiment as experiment
    import libcity.pipeline.runtime_tuning as tuning
    atomic_fixture(tmp_path/'raw',steps=80)
    cfg=tiny_config(tmp_path,'study',epochs=1)
    actual_run=experiment.run_experiment
    actual_prepare=experiment.prepare_traffic
    active={'mode':None}
    seen=[]
    test_evaluations=[]
    def prepare(data,training,evaluation,seed,logger=None):
        bundle=actual_prepare(data,training,evaluation,seed,logger)
        seen.append((data['input_window'],data['scaler']['name'],len(bundle.train.dataset)))
        original=bundle.test
        class Guard:
            def __iter__(self):
                assert active['mode']=='evaluate','A tuning trial touched the held-out test loader'
                test_evaluations.append(True)
                return iter(original)
        bundle.test=Guard()
        return bundle
    def run(mode,*args,**kwargs):
        active['mode']=mode
        if mode=='train':
            assert kwargs['evaluate_test'] is False
        return actual_run(mode,*args,**kwargs)
    monkeypatch.setattr(experiment,'prepare_traffic',prepare)
    monkeypatch.setattr(tuning,'run_experiment',run)
    result=run_search(config=cfg,space={'data.input_window':[4,6],'data.scaler.name':['none','standard']},max_trials=4)
    report=json.loads(result.artifacts['search_report'].read_text())
    assert report['test_evaluations']==len(test_evaluations)==1
    assert set((a,b) for a,b,_ in seen[:4])=={(4,'none'),(4,'standard'),(6,'none'),(6,'standard')}
    assert len({count for _,_,count in seen[:4]})==2
    assert report['best_validation']==min(trial['validation_score'] for trial in report['trials'])
    for trial in report['trials']:
        folder=Path(trial['resolved_config']).parent
        assert not (folder/'eval').exists()
        assert 'test_metrics' not in trial
    assert len({trial['resolved_config'] for trial in report['trials']})==4


def test_search_respects_maximizing_recall_and_seeded_random_choices(tmp_path):
    task_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,'RNN','rank-search')
    cfg['training']['epochs']=1
    result=run_search(config=cfg,space={'model.params.hidden_size':[4,8]},max_trials=2)
    report=json.loads(result.artifacts['search_report'].read_text())
    assert report['direction']=='max'
    assert report['best_validation']==max(trial['validation_score'] for trial in report['trials'])
    domains={'training.optimizer.lr':[.001,.01],'data.input_window':[4,6,8]}
    first=candidates(domains,'random',4,17)
    assert first==candidates(domains,'random',4,17)
    assert len(first)==len({json.dumps(item,sort_keys=True) for item in first})==4


@pytest.mark.parametrize('space',[{}, {'bad':[1]}, {'training.epochs':[]}, {'training.typo':[1]}, {'runtime.seed':[2]}])
def test_invalid_search_fails_before_creating_study(tmp_path,space):
    cfg=tiny_config(tmp_path,'invalid',epochs=1)
    with pytest.raises((ValueError,TypeError)):
        run_search(config=cfg,space=space)
    assert not (tmp_path/'runs/invalid').exists()


def test_new_hyper_cli_runs_outside_repository(tmp_path):
    atomic_fixture(tmp_path/'raw')
    cfg=tiny_config(tmp_path,'cli-search',epochs=1)
    config_file=tmp_path/'gru.json'
    config_file.write_text(json.dumps(cfg))
    search=tmp_path/'space.json'
    search.write_text(json.dumps({'training.optimizer.lr':[.001,.002]}))
    repo=Path(__file__).resolve().parents[1]
    process=subprocess.run([sys.executable,'-B',str(repo/'run_hyper.py'),'--config',str(config_file),
        '--space',str(search),'--max-trials','2'],cwd=tmp_path.parent,text=True,capture_output=True)
    assert process.returncode==0,process.stderr
    result=json.loads(process.stdout.strip().splitlines()[-1])
    report=json.loads(Path(result['artifacts']['search_report']).read_text())
    assert len(report['trials'])==2 and report['test_evaluations']==1
