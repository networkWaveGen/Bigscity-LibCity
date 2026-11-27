"""Resume rejects changed data even when metadata and normalization still match."""
import json
import os
import shutil

import pandas as pd
import pytest
import torch

from libcity.data.sources import atomic_sources
from libcity.executor.train import train_loop
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


@pytest.mark.parametrize('changed',['dyna','rel'])
def test_changed_values_or_graph_are_rejected_before_any_update(tmp_path,monkeypatch,changed):
    folder = atomic_fixture(tmp_path/'raw',steps=70)
    def config(experiment,epochs):
        value = tiny_config(tmp_path,experiment,epochs=epochs,batch_size=8)
        if changed=='rel':
            value['model'] = dict(name='STGCN',params=dict(Ks=2,Kt=2,blocks=[[1,4,4],[4,4,4]],dropout=.1))
            value['data']['input_window'] = 12
        return value
    first = run_experiment('train',config=config('first',1))
    payload = torch.load(first.artifacts['last'],weights_only=True)
    assert payload['format_version']==3
    assert all(set(row)=={'name','size','mtime_ns'} for row in payload['atomic_sources'])
    path = folder/('Tiny.'+changed)
    frame = pd.read_csv(path)
    if changed=='dyna':
        frame['speed'] += 1000
    else:
        frame['destination_id'] = (frame['destination_id']+1)%3
    frame.to_csv(path,index=False)
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *_args,**_kwargs:pytest.fail('Changed data must fail before optimizer update'))
    with pytest.raises(ValueError,match='atomic source files changed'):
        run_experiment('train',config=config('changed',2),resume=first.artifacts['last'])
    assert not (tmp_path/'runs/changed/checkpoints').exists()


def test_mtime_change_is_conservative_even_when_file_size_is_unchanged(tmp_path):
    folder = atomic_fixture(tmp_path/'raw')
    first = run_experiment('train',config=tiny_config(tmp_path,'first',epochs=1,batch_size=8))
    before = atomic_sources(dict(data_dir=str(tmp_path/'raw'),dataset='Tiny'))
    path = folder/'Tiny.dyna';stat = path.stat()
    os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+2_000_000_000))
    after = atomic_sources(dict(data_dir=str(tmp_path/'raw'),dataset='Tiny'))
    assert [row['size'] for row in before]==[row['size'] for row in after]
    assert before!=after
    with pytest.raises(ValueError,match='modification times'):
        run_experiment('train',config=tiny_config(tmp_path,'changed',epochs=2,batch_size=8),resume=first.artifacts['last'])


def test_preserved_sources_can_move_and_cache_changes_do_not_break_exact_resume(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=70)
    def config(experiment,epochs):
        value = tiny_config(tmp_path,experiment,epochs=epochs,batch_size=8,shuffle=True)
        value['data']['cache'] = True
        return value
    full = run_experiment('train',config=config('full',2))
    first = run_experiment('train',config=config('first',1))
    moved = tmp_path/'moved/raw'
    shutil.copytree(tmp_path/'raw',moved,copy_function=shutil.copy2)
    resumed_config = config('moved',2)
    resumed_config['data'].update(data_dir=str(moved),cache_dir=str(tmp_path/'moved/cache'))
    resumed = run_experiment('train',config=resumed_config,resume=first.artifacts['last'])
    expected = torch.load(full.artifacts['last'],weights_only=True)
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert actual['atomic_sources']==expected['atomic_sources']
    assert actual['train_state']==expected['train_state']
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)


def test_nested_and_external_references_are_recorded_without_generated_outputs(tmp_path):
    folder = atomic_fixture(tmp_path/'raw')
    external = tmp_path/'shared';external.mkdir()
    (external/'roads.geo').write_text('geo_id\n7\n')
    (external/'auxiliary.txt').write_text('observation\n')
    path = folder/'config.json';metadata = json.loads(path.read_text())
    metadata.update(geo_file='../../shared/roads',auxiliary_file='../../shared/auxiliary.txt')
    path.write_text(json.dumps(metadata))
    nested = folder/'nested';nested.mkdir()
    (nested/'more.dyna').write_text('dyna_id\n1\n')
    (folder/'generated.npz').write_bytes(b'not an atomic input')
    records = atomic_sources(dict(data_dir=str(tmp_path/'raw'),dataset='Tiny'))
    names = {row['name'] for row in records}
    assert str((external/'roads.geo').resolve()) in names
    assert str((external/'auxiliary.txt').resolve()) in names
    assert 'nested/more.dyna' in names and 'generated.npz' not in names


def test_old_full_format_is_rejected_and_pure_weights_still_evaluate(tmp_path):
    atomic_fixture(tmp_path/'raw')
    first = run_experiment('train',config=tiny_config(tmp_path,'first',epochs=1,batch_size=8))
    payload = torch.load(first.artifacts['last'],weights_only=True)
    payload['format_version'] = 2
    payload.pop('atomic_sources')
    old = tmp_path/'old.pt';torch.save(payload,old)
    with pytest.raises(ValueError,match='format 3'):
        run_experiment('train',config=tiny_config(tmp_path,'old',epochs=2,batch_size=8),resume=old)
    evaluated = run_experiment('evaluate',config=tiny_config(tmp_path,'evaluate',epochs=1,batch_size=8),checkpoint=first.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in first.metrics.items() if key!='best_validation'}


def test_checkpoint_keeps_the_source_version_from_context_creation(tmp_path):
    folder = atomic_fixture(tmp_path/'raw')
    context = make_context(tiny_config(tmp_path,'in-memory',epochs=2,batch_size=8))
    try:
        snapshot = context.atomic_sources
        train_loop(context,until_epoch=1)
        frame = pd.read_csv(folder/'Tiny.dyna');frame['speed'] += 10
        frame.to_csv(folder/'Tiny.dyna',index=False)
        train_loop(context)
        checkpoint = context.directory/'checkpoints/last.pt'
        assert torch.load(checkpoint,weights_only=True)['atomic_sources']==snapshot
    finally:
        context.tracker.close()
    with pytest.raises(ValueError,match='atomic source files changed'):
        run_experiment('train',config=tiny_config(tmp_path,'changed',epochs=3,batch_size=8),resume=checkpoint)
