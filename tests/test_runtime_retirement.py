"""Replacement contracts for deleted configuration, batch and checkpoint APIs."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from libcity.config.runtime import load_config
from libcity.data.list_dataset import ListDataset
from libcity.data.utils import TensorCollator,generate_dataloader,generate_dataloader_pad
from libcity.pipeline.experiment import run_experiment
from libcity.utils.runtime_checkpoint import CheckpointManager
from libcity.testing.fixtures import task_fixture
from libcity.testing.fixtures import atomic_fixture
from runtime_helpers import make_context
from test_runtime_tasks import configuration


def test_retired_modules_and_executor_classes_are_absent():
    for name in ('libcity.registry','libcity.config.config_parser','libcity.data.batch',
                 'libcity.pipeline.pipeline','libcity.pipeline.tuning','libcity.executor.traffic_state_executor',
                 'libcity.executor.traj_loc_pred_executor','libcity.executor.hyper_tuning'):
        assert importlib.util.find_spec(name) is None,name
    root=Path(__file__).resolve().parents[1]
    for path in (root/'libcity/executor').glob('*.py'):
        assert not any(isinstance(node,ast.ClassDef) and node.name.endswith('Executor')
                       for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))))
    for path in (root/'libcity').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node,ast.ImportFrom):
                assert node.module not in ('libcity.registry','libcity.data.batch','libcity.config.config_parser')
    for name in ('test_model.py','unit_test.py','hyper_tune.py','tools/migrate_checkpoint.py'):
        assert not (root/name).exists()


def test_tensor_collation_owns_values_and_padding_keeps_explicit_lengths():
    samples=[([1,2,3,4],np.array([1.,2.])),([8,9],np.array([3.,4.]))]
    before=copy.deepcopy(samples)
    loader=generate_dataloader_pad(samples,samples,samples,{'history':'int','y':'float'},2,0,
                                  pad_item={'history':0},pad_max_len={'history':3},shuffle=False)[0]
    batch=next(iter(loader))
    assert type(batch) is dict and all(isinstance(v,torch.Tensor) for v in batch.values())
    assert batch['history'].tolist()==[[2,3,4],[8,9,0]]
    assert batch['history_lengths'].tolist()==[3,2]
    batch['y'].fill_(99)
    assert samples[0][0]==before[0][0]
    np.testing.assert_array_equal(samples[0][1],before[0][1])
    with pytest.raises(ValueError,match='fields'):
        TensorCollator({'X':'float'})([(1,2)])


def test_tensor_loader_spawn_and_heterogeneous_tail_padding():
    samples=[(np.ones((4,2,1))*i,np.ones((2,2,1))*i) for i in range(3)]
    loaders=generate_dataloader(samples,samples,samples,{'X':'float','y':'float'},2,0,shuffle=False,pad_with_last_sample=True)
    assert [sum(len(batch['X']) for batch in loader) for loader in loaders]==[4,3,3]
    assert next(iter(loaders[0]))['X'].shape==(2,4,2,1)
    worker=DataLoader(ListDataset(samples),batch_size=2,num_workers=1,multiprocessing_context='spawn',
                      collate_fn=TensorCollator({'X':'float','y':'float'}))
    assert sum(len(batch['X']) for batch in worker)==3


@pytest.mark.parametrize('extension',['.m','.tar','.pth'])
def test_legacy_checkpoint_extensions_fail_before_preparing_data(tmp_path,extension):
    path=tmp_path/('old'+extension)
    torch.save(torch.nn.Linear(1,1).state_dict(),path)
    cfg=dict(model={'name':'GRU'},data={'dataset':'does-not-exist','data_dir':str(tmp_path)},
             runtime={'output_dir':str(tmp_path/'runs')})
    with pytest.raises(ValueError,match=r'\.pt'):
        run_experiment('evaluate',config=cfg,checkpoint=path)
    assert not (tmp_path/'runs').exists()


@pytest.mark.parametrize('format_name',['pair','epoch','state_dict','object'])
def test_new_loader_rejects_historical_formats_without_unpickling_fallback(tmp_path,format_name):
    model=torch.nn.Linear(1,1)
    state=model.state_dict()
    payload={'pair':(state,{}),'epoch':dict(model_state_dict=state,optimizer_state_dict={},epoch=1),
             'state_dict':{'state_dict':state},'object':model}[format_name]
    path=tmp_path/'weights.pt'
    torch.save(payload,path)
    with pytest.raises((ValueError,pickle.UnpicklingError)):
        CheckpointManager.load_weights(path,model)


def test_flat_configs_and_training_mode_aliases_are_not_silently_accepted(tmp_path):
    with pytest.raises((ValueError,TypeError)):
        load_config(config=dict(task='traffic_state_pred',model='GRU',dataset='Tiny',train=True))


def test_native_eta_export_keeps_repeated_users_and_all_tail_trajectories(tmp_path):
    task_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,'DeepTTE','export')
    cfg['training']['epochs']=1
    cfg['evaluation']={'batch_size':3}
    context=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'inspect'}})
    expected={int(row['sample_id']) for row in context.bundle.test.dataset.rows}
    context.tracker.close()
    result=run_experiment('train',config=cfg)
    records=json.loads(result.artifacts['predictions'].read_text())
    assert {row['sample_id'] for row in records}==expected
    assert len(records)==len(expected)>1
    assert len({row['traj_id'] for row in records})==len(expected)


def test_visualization_cli_uses_explicit_paths_from_outside_source(tmp_path):
    atomic_fixture(tmp_path/'data')
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/'geojson'
    process=subprocess.run([sys.executable,'-B',str(root/'visualize.py'),'--dataset','Tiny',
        '--data-dir',str(tmp_path/'data'),'--output-dir',str(output)],cwd=tmp_path.parent,
        capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    records=json.loads(next(output.glob('*.json')).read_text())
    assert len(records['features'])==3
    assert records['features'][0]['geometry']['coordinates']==[0,0]
