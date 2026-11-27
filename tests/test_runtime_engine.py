import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from libcity.evaluator.runtime_metrics import ForecastMetrics
from libcity.executor.components import build_optimizer
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.executor.evaluate import ExperimentResult
from libcity.executor.evaluate import StepResult,supervised_step
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from runtime_helpers import make_context, tiny_config

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


@pytest.fixture
def data(tmp_path):
    atomic_fixture(tmp_path / 'raw')
    return tmp_path


def close(*contexts):
    for context in contexts:
        context.tracker.close()


def test_epoch_schedule_waits_for_real_updates_and_resumes_after_empty_epochs(data,recwarn):
    scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.5)
    def run(context,until_epoch=None):
        standard=supervised_step(context.loss_fn)
        def step(model,batch,state):
            if state.epoch<2:
                return StepResult(model(context.loss_fn.inputs(batch)).sum()*0.,0)
            return standard(model,batch,state)
        return train_loop(context,step,until_epoch=until_epoch)
    full=make_context(tiny_config(data,'empty-full',epochs=3,batch_size=8,scheduler=scheduler))
    partial=make_context(tiny_config(data,'empty-partial',epochs=3,batch_size=8,scheduler=scheduler))
    try:
        run(full)
        run(partial,until_epoch=2)
        assert partial.state.optimizer_steps==0 and partial.schedule.scheduler.last_epoch==0
        assert partial.optimizer.param_groups[0]['lr']==.001
        saved=partial.directory/'checkpoints/last.pt'
        resumed=make_context(tiny_config(data,'empty-resumed',epochs=3,batch_size=8,scheduler=scheduler))
        try:
            resumed.checkpoints.load_last(saved,resumed)
            run(resumed)
            assert resumed.schedule.scheduler.last_epoch==full.schedule.scheduler.last_epoch==1
            assert resumed.optimizer.param_groups[0]['lr']==full.optimizer.param_groups[0]['lr']==.0005
            expected=torch.load(full.directory/'checkpoints/last.pt',weights_only=True)
            actual=torch.load(resumed.directory/'checkpoints/last.pt',weights_only=True)
            assert expected['train_state']==actual['train_state'] and expected['scheduler']==actual['scheduler']
            for name,value in expected['model'].items():
                torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
        finally:
            close(resumed)
        assert not any('lr_scheduler.step()' in str(item.message) for item in recwarn)
    finally:
        close(full,partial)


def test_gru_cli_python_train_evaluate_and_resume(data, monkeypatch):
    config = tiny_config(data, epochs=1)
    config_file = data / 'experiment.json'
    config_file.write_text(json.dumps(config))
    result = run_experiment('train', config_file)
    cmd = [sys.executable, '-B', str(ROOT/'run_model.py'), 'train', '--config', str(config_file),
           '--set', 'runtime.exp_id=cli']
    command = subprocess.run(cmd, cwd=data.parent, text=True, capture_output=True, check=True)
    cli = json.loads(command.stdout.splitlines()[-1])
    assert cli['metrics'] == result.metrics
    with np.load(result.artifacts['predictions']) as api, np.load(cli['artifacts']['predictions']) as other:
        np.testing.assert_array_equal(api['prediction'], other['prediction'])
    weights = torch.load(result.artifacts['best'], weights_only=True)
    assert weights and all(isinstance(value, torch.Tensor) for value in weights.values())
    last = torch.load(result.artifacts['last'], weights_only=True)
    assert {'optimizer', 'scheduler', 'rng', 'early_stopper', 'amp_scaler', 'train_state'} <= last.keys()
    # Evaluation is a distinct construction path, without training objects.
    import libcity.pipeline.experiment as pipeline
    monkeypatch.setattr(pipeline, 'build_context', lambda *_: pytest.fail('Evaluation initialized training'))
    evaluated = run_experiment('evaluate', config_file, checkpoint=result.artifacts['best'],
        overrides={'runtime': {'exp_id': 'evaluation'}})
    for key, value in evaluated.metrics.items():
        assert value == pytest.approx(result.metrics[key])
    assert not (data/'runs/evaluation/metrics.jsonl').exists()
    assert not (data/'runs/evaluation/tensorboard').exists()
    monkeypatch.undo()
    resumed = subprocess.run([sys.executable, '-B', str(ROOT/'run_model.py'), 'train', '--config', str(config_file),
        '--resume', str(result.artifacts['last']), '--set', 'training.epochs=2', '--set', 'runtime.exp_id=resumed'],
        cwd=data.parent, text=True, capture_output=True, check=True)
    resumed = json.loads(resumed.stdout.splitlines()[-1])
    assert torch.load(resumed['artifacts']['last'], weights_only=True)['train_state']['epoch'] == 2
    evaluation = subprocess.run([sys.executable, '-B', str(ROOT/'run_model.py'), 'evaluate', '--config', str(config_file),
        '--checkpoint', str(result.artifacts['best']), '--set', 'runtime.exp_id=cli-evaluation',
        '--set', 'evaluation.save_predictions=false'], cwd=data.parent, text=True, capture_output=True, check=True)
    evaluation = json.loads(evaluation.stdout.splitlines()[-1])
    assert 'predictions' not in evaluation['artifacts']


@pytest.mark.parametrize('scheduler', [
    {'name':'step', 'interval':'step', 'step_size':2, 'gamma':.8},
    {'name':'step', 'interval':'epoch', 'step_size':1, 'gamma':.8},
    {'name':'plateau', 'interval':'metric', 'patience':0, 'factor':.8, 'mode':'max'},
    {'name':'onecycle', 'interval':'step', 'max_lr':.01},
    {'name':'warmup_cosine', 'interval':'step', 'warmup_steps':2},
])
def test_complete_resume_matches_next_updates_lr_counters_and_best(data, scheduler):
    cfg = tiny_config(data, 'uninterrupted', scheduler=scheduler, accumulation_steps=3)
    uninterrupted = make_context(cfg)
    train_loop(uninterrupted)
    expected = torch.load(uninterrupted.directory/'checkpoints/last.pt', weights_only=True)
    first = make_context({**cfg, 'runtime': {**cfg['runtime'], 'exp_id': 'first'}})
    train_loop(first, until_epoch=1)
    resumed = make_context({**cfg, 'runtime': {**cfg['runtime'], 'exp_id': 'resumed'}})
    resumed.checkpoints.load_last(first.directory/'checkpoints/last.pt', resumed)
    assert resumed.state.epoch == 1 and resumed.state.batches_seen == len(first.bundle.train)
    train_loop(resumed)
    actual = torch.load(resumed.directory/'checkpoints/last.pt', weights_only=True)
    assert actual['train_state'] == expected['train_state']
    assert actual['scheduler'] == expected['scheduler']
    assert actual['optimizer']['param_groups'] == expected['optimizer']['param_groups']
    assert actual['early_stopper'] == expected['early_stopper']
    for key in expected['model']:
        torch.testing.assert_close(actual['model'][key], expected['model'][key], rtol=0, atol=0)
        torch.testing.assert_close(actual['best_model'][key], expected['best_model'][key], rtol=0, atol=0)
    for key, expected_state in expected['optimizer']['state'].items():
        for name, value in expected_state.items():
            torch.testing.assert_close(actual['optimizer']['state'][key][name], value, rtol=0, atol=0)
    close(uninterrupted, first, resumed)


def test_accumulation_matches_combined_batches_and_flushes_short_tail(data):
    cfg = tiny_config(data, 'accumulated', epochs=1, accumulation_steps=3,
        optimizer={'name':'sgd', 'lr':.001}, loss={'name':'masked_mse'})
    context = make_context(cfg)
    initial = copy.deepcopy(context.model.state_dict())
    reference = copy.deepcopy(context.model)
    reference.load_state_dict(initial)
    optimizer = torch.optim.SGD(reference.parameters(), lr=.001)
    batches = list(context.bundle.train)
    reference.train()
    for offset in range(0, len(batches), 3):
        combined = {key: torch.cat([batch[key] for batch in batches[offset:offset+3]]) for key in batches[0]}
        optimizer.zero_grad()
        loss = context.loss_fn(reference(model_inputs(combined, 1)), combined['y'])
        loss.loss.backward()
        optimizer.step()
    train_loop(context)
    actual = torch.load(context.directory/'checkpoints/last.pt', weights_only=True)
    assert context.state.optimizer_steps == (len(batches) + 2)//3
    for key in initial:
        torch.testing.assert_close(actual['model'][key], reference.state_dict()[key], rtol=1e-5, atol=1e-6)
    close(context)


@pytest.mark.parametrize('mode', ['single', 'average'])
def test_global_metrics_with_tail_missing_mask_and_multiple_horizons(mode):
    target = torch.tensor([0., 2., 1., float('nan'), 3., 4.]).reshape(3,2,1,1)
    predicted = torch.tensor([9., 1., 2., 4., 5., 7.]).reshape(3,2,1,1)
    names = ['MAE','MSE','RMSE','MAPE','R2','EVAR']
    together = ForecastMetrics(names, 0., mode)
    together.update(predicted, target)
    batches = ForecastMetrics(names, 0., mode)
    batches.update(predicted[:2], target[:2])
    batches.update(predicted[2:], target[2:])
    assert batches.compute() == pytest.approx(together.compute())
    assert batches.compute()['MAE@1'] == pytest.approx(1.5)
    expected = 2. if mode == 'single' else 1.75
    assert batches.compute()['MAE@2'] == pytest.approx(expected)


def test_optimizer_groups_freeze_and_reject_overlap():
    model = torch.nn.Sequential(torch.nn.Linear(2,2), torch.nn.Linear(2,1))
    optimizer = build_optimizer(model, dict(name='adam', lr=.01, groups=[
        {'modules':['0'], 'freeze':True}, {'modules':['1'], 'lr':.1, 'weight_decay':.2}]))
    assert not model[0].weight.requires_grad
    assert optimizer.param_groups[0]['lr'] == .1
    assert optimizer.param_groups[0]['weight_decay'] == .2
    with pytest.raises(ValueError, match='overlap'):
        build_optimizer(model, dict(name='sgd', lr=.1, groups=[{'modules':['1']}, {'modules':['1']}]))


def test_checkpoint_errors_and_optional_import_isolation(data):
    config = tiny_config(data)
    with pytest.raises(ValueError, match='requires --checkpoint'):
        run_experiment('evaluate', config=config)
    with pytest.raises(FileNotFoundError, match='does not exist'):
        run_experiment('evaluate', config=config, checkpoint=data/'missing.pt')
    assert not (data/'runs').exists()
    command = subprocess.run([sys.executable,'-B','-c',
        "import sys; from libcity.pipeline.experiment import run_experiment; "
        "assert not {'ray','hyperopt','dgl','torchtext','tensorboard','torch.utils.tensorboard'} & set(sys.modules)"],
        cwd=ROOT, capture_output=True, text=True)
    assert command.returncode == 0, command.stderr


def test_best_weights_are_not_resume_state_and_changed_settings_fail(data):
    context = make_context(tiny_config(data, epochs=1))
    train_loop(context)
    with pytest.raises(ValueError, match='complete last.pt'):
        context.checkpoints.load_last(context.directory/'checkpoints/best.pt', context)
    other = make_context(tiny_config(data, 'changed', epochs=1, accumulation_steps=2))
    with pytest.raises(ValueError, match='configuration'):
        other.checkpoints.load_last(context.directory/'checkpoints/last.pt', other)
    close(context, other)


@pytest.mark.parametrize('direction,monitor,values', [
    ('min', 'val_loss', [3., 1., 2.]), ('max', 'val_Recall@1', [1., 3., 2.]),
])
def test_best_and_last_remain_distinct_for_min_and_max(data, monkeypatch, direction, monitor, values):
    context = make_context(tiny_config(data, direction=direction, monitor=monitor))
    snapshots = []
    def validate(model, *_args, **_kwargs):
        snapshots.append(copy.deepcopy(model.state_dict()))
        return ExperimentResult({monitor: values[len(snapshots)-1]})
    monkeypatch.setattr('libcity.executor.train.evaluate_loop', validate)
    train_loop(context)
    last = torch.load(context.directory/'checkpoints/last.pt', weights_only=True)
    best = torch.load(context.directory/'checkpoints/best.pt', weights_only=True)
    for key in best:
        torch.testing.assert_close(best[key], snapshots[1][key], rtol=0, atol=0)
        torch.testing.assert_close(last['model'][key], snapshots[2][key], rtol=0, atol=0)
    assert last['early_stopper']['wait'] == 1
    close(context)


def test_cpu_amp_and_opt_in_tensorboard(data):
    config = tiny_config(data, epochs=1, amp=True)
    config['runtime']['tensorboard'] = True
    result = run_experiment('train', config=config)
    assert np.isfinite(list(result.metrics.values())).all()
    assert list((data/'runs/run/tensorboard').glob('events.*'))
    state = torch.load(result.artifacts['last'], weights_only=True)
    assert state['train_state']['optimizer_steps'] > 0


def test_early_stop_and_completed_resume_perform_no_extra_update(data,monkeypatch):
    cfg=tiny_config(data,epochs=5,early_stopping={'patience':1,'min_delta':0.})
    context=make_context(cfg)
    scores=iter([3.,1.,2.])
    monkeypatch.setattr('libcity.executor.train.evaluate_loop',
                        lambda *args,**kwargs:ExperimentResult({'val_loss':next(scores)}))
    train_loop(context)
    assert context.state.epoch==3 and context.stopper.stopped
    saved=torch.load(context.directory/'checkpoints/last.pt',weights_only=True)
    resumed=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'resumed'}})
    resumed.checkpoints.load_last(context.directory/'checkpoints/last.pt',resumed)
    train_loop(resumed)
    assert resumed.state.optimizer_steps==saved['train_state']['optimizer_steps']
    assert (resumed.directory/'checkpoints/last.pt').exists()
    close(context,resumed)


def test_dropout_and_shuffled_loader_rng_resume_exactly(data):
    cfg=tiny_config(data,epochs=2,shuffle=True)
    cfg['model']['params'].update(num_layers=2,dropout=.2)
    full=make_context(cfg)
    train_loop(full)
    expected=torch.load(full.directory/'checkpoints/last.pt',weights_only=True)
    partial=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'partial'}})
    train_loop(partial,until_epoch=1)
    resumed=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'resumed'}})
    resumed.checkpoints.load_last(partial.directory/'checkpoints/last.pt',resumed)
    train_loop(resumed)
    actual=torch.load(resumed.directory/'checkpoints/last.pt',weights_only=True)
    for key in actual['model']:
        torch.testing.assert_close(actual['model'][key],expected['model'][key],rtol=0,atol=0)
    torch.testing.assert_close(actual['loader_rng'],expected['loader_rng'],rtol=0,atol=0)
    assert actual['train_state']==expected['train_state']
    close(full,partial,resumed)


def test_clip_is_applied_after_objective_normalization(data):
    context=make_context(tiny_config(data,epochs=1,batch_size=128,clip_grad_norm=.05,
                                    optimizer={'name':'sgd','lr':.1}))
    reference=copy.deepcopy(context.model)
    batch=next(iter(context.bundle.train))
    loss=context.loss_fn(reference(model_inputs(batch,1)),batch['y'])
    loss.loss.backward()
    norm=torch.stack([param.grad.norm() for param in reference.parameters()]).norm()
    assert norm>.05
    with torch.no_grad():
        for param in reference.parameters():
            param.sub_(.1*param.grad*min(1.,.05/(norm.item()+1e-6)))
    train_loop(context)
    state=torch.load(context.directory/'checkpoints/last.pt',weights_only=True)
    for key,value in reference.state_dict().items():
        torch.testing.assert_close(state['model'][key],value,rtol=1e-6,atol=1e-7)
    close(context)


def test_resume_rejects_changed_readonly_source_metadata(data):
    context=make_context(tiny_config(data,epochs=1))
    train_loop(context)
    # Keep experiment config and tensor shapes identical, but choose a different
    # intrinsic source column through the dataset's own config.json.
    import pandas as pd
    path=data/'raw/Tiny/Tiny.dyna'
    frame=pd.read_csv(path)
    frame['flow']=frame['speed']+100
    frame.to_csv(path,index=False)
    path=data/'raw/Tiny/config.json'
    metadata=json.loads(path.read_text())
    metadata['data_col']=['flow']
    path.write_text(json.dumps(metadata))
    changed=make_context(tiny_config(data,'changed',epochs=1))
    with pytest.raises(ValueError,match='metadata/normalization'):
        changed.checkpoints.load_last(context.directory/'checkpoints/last.pt',changed)
    close(context,changed)
