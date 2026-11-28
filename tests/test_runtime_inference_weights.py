"""Standalone tensor weights must retain the meaning of their input metadata."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.data.od_bundle import prepare_od
from libcity.executor.train import train_loop
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,od_fixture
from libcity.utils.checkpoint import copy_state_to_cpu
from libcity.utils.normalization import StandardScaler
from libcity.utils.runtime_checkpoint import CheckpointManager,inference_spec,weight_spec
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def cstn_config(root,name):
    return dict(model=dict(name='CSTN',params=dict(n_layers=1,c_lt=4)),
        data=dict(dataset='ODTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=2,pad_training=False,features=dict(known_future_columns=['holiday'])),
        training=dict(epochs=1,batch_size=5),evaluation=dict(batch_size=4),
        runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def test_safe_standalone_state_dict_keeps_tensor_keys_module_versions_and_owned_input_metadata(tmp_path):
    atomic_fixture(tmp_path/'raw')
    context = make_context(tiny_config(tmp_path,epochs=1))
    try:
        train_loop(context)
        state = torch.load(context.directory/'checkpoints/best.pt',weights_only=True)
        assert isinstance(state,OrderedDict)
        assert state.keys()==context.model.state_dict().keys()
        assert all(isinstance(value,torch.Tensor) for value in state.values())
        assert state._metadata['']['version']==context.model._version
        assert weight_spec(state)==inference_spec(context.bundle,context.config)
        json.dumps(weight_spec(state),allow_nan=False)
        cloned = copy_state_to_cpu(state)
        assert cloned._metadata==state._metadata
        cloned._metadata['']['libcity_input_spec']['model']['name']='changed'
        assert weight_spec(state)['model']['name']=='GRU'
        # Normal PyTorch loading still accepts this pure state_dict.
        model = deepcopy(context.model)
        model.load_state_dict(state)
        standalone = tmp_path/'standalone';standalone.mkdir()
        copied = standalone/'copied.pt'
        shutil.copyfile(context.directory/'checkpoints/best.pt',copied)
        assert list(standalone.iterdir())==[copied]
        CheckpointManager.load_weights(copied,model,input_spec=inference_spec(context.bundle,context.config))
    finally:
        context.tracker.close()


@pytest.mark.parametrize('change',['scaler','node_ids','grid_positions','future_order','history_order','input_window','model_params'])
def test_same_shape_incompatible_inputs_are_rejected_before_any_model_buffer_is_loaded(tmp_path,change,monkeypatch):
    od_fixture(tmp_path/'raw',features=1,grid_shape=(2,2))
    context = make_context(cstn_config(tmp_path,'saved'))
    try:
        path = context.checkpoints.save_best(context.model)
        bundle,cfg = context.bundle,deepcopy(context.config)
        meta = bundle.metadata
        layout,info = dict(meta.input_layout),dict(meta.info)
        if change=='scaler':
            bundle = replace(bundle,scaler=StandardScaler(20.,2.))
        elif change=='node_ids':
            info['node_ids'] = list(reversed(info['node_ids']))
        elif change=='grid_positions':
            layout['grid_positions'] = list(reversed(layout['grid_positions']))
        elif change=='future_order':
            layout['future_feature_names'] = list(reversed(layout['future_feature_names']))
        elif change=='history_order':
            layout['history_features'] = dict(layout['history_features'],columns=list(reversed(layout['history_features']['columns'])))
        elif change=='input_window':
            cfg['data']['input_window'] += 1
        elif change=='model_params':
            cfg['model']['params']['n_layers'] += 1
        if change not in ('scaler','input_window','model_params'):
            bundle = replace(bundle,metadata=replace(meta,info=info,input_layout=layout))
        monkeypatch.setattr(context.model,'load_state_dict',lambda *_:pytest.fail('Input rejection must precede tensor loading'))
        with pytest.raises(ValueError,match='input specification does not match'):
            CheckpointManager.load_weights(path,context.model,input_spec=inference_spec(bundle,cfg))
    finally:
        context.tracker.close()


def test_pipeline_evaluates_copied_weights_with_new_batches_without_training_components_and_rejects_new_scaler(tmp_path,monkeypatch):
    atomic_fixture(tmp_path/'raw')
    cfg = tiny_config(tmp_path,'trained',epochs=1)
    trained = run_experiment('train',config=cfg)
    copied = tmp_path/'independent-best.pt'
    shutil.copyfile(trained.artifacts['best'],copied)
    # Only the standalone weight file is supplied to the public evaluator.
    evaluation = deepcopy(cfg)
    evaluation['runtime']['exp_id']='evaluated'
    evaluation['training']['epochs']=4
    evaluation['training']['batch_size']=7
    evaluation['data']['pad_training']=True
    evaluation['evaluation']=dict(batch_size=3)
    import libcity.executor.train as training
    monkeypatch.setattr(training,'build_optimizer',lambda *_:pytest.fail('Evaluation initialized an optimizer'))
    monkeypatch.setattr(training,'MetricTracker',lambda *_:pytest.fail('Evaluation initialized a training recorder'))
    result = run_experiment('evaluate',config=evaluation,checkpoint=copied)
    with np.load(trained.artifacts['predictions'],allow_pickle=False) as left,np.load(result.artifacts['predictions'],allow_pickle=False) as right:
        np.testing.assert_allclose(left['prediction'],right['prediction'],rtol=2e-6,atol=2e-6)
        np.testing.assert_equal(left['truth'],right['truth'])
    bad = deepcopy(evaluation)
    bad['runtime']['exp_id']='bad-scaler'
    bad['data']['scaler']=dict(name='standard')
    with pytest.raises(ValueError,match='input specification does not match'):
        run_experiment('evaluate',config=bad,checkpoint=copied)


def test_public_pipeline_rejects_weights_without_specification(tmp_path):
    atomic_fixture(tmp_path/'raw')
    context = make_context(tiny_config(tmp_path,'snapshot',epochs=1))
    try:
        path = tmp_path/'bare.pt'
        torch.save(context.model.state_dict(),path)
    finally:
        context.tracker.close()
    with pytest.raises(ValueError,match='lack an input specification'):
        run_experiment('evaluate',config=tiny_config(tmp_path,'bare'),checkpoint=path)


def test_completed_resume_validates_legacy_last_and_reexports_standalone_best_without_an_extra_update(tmp_path):
    atomic_fixture(tmp_path/'raw')
    context = make_context(tiny_config(tmp_path,'partial',epochs=1))
    try:
        train_loop(context)
        payload = torch.load(context.directory/'checkpoints/last.pt',weights_only=True)
        del payload['best_model']._metadata['']['libcity_input_spec']
        legacy = tmp_path/'legacy-last.pt';torch.save(payload,legacy)
    finally:
        context.tracker.close()
    exported = run_experiment('train',config=tiny_config(tmp_path,'exported',epochs=1),resume=legacy)
    last = torch.load(exported.artifacts['last'],weights_only=True)
    assert last['train_state']==payload['train_state']
    assert weight_spec(last['best_model']) is not None
    assert weight_spec(torch.load(exported.artifacts['best'],weights_only=True))==weight_spec(last['best_model'])
    evaluated = run_experiment('evaluate',config=tiny_config(tmp_path,'evaluated',epochs=1),checkpoint=exported.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in exported.metrics.items() if key!='best_validation'}


def test_resume_rejects_conflicting_best_specification_before_state_restore(tmp_path):
    atomic_fixture(tmp_path/'raw')
    context = make_context(tiny_config(tmp_path,'saved',epochs=1))
    target = None
    try:
        train_loop(context)
        payload = torch.load(context.directory/'checkpoints/last.pt',weights_only=True)
        payload['best_model']._metadata['']['libcity_input_spec']['model']['name']='wrong'
        path = tmp_path/'conflicting-last.pt';torch.save(payload,path)
        target = make_context(tiny_config(tmp_path,'target',epochs=1))
        before = copy_state_to_cpu(target.model.state_dict())
        rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError,match='input specification does not match'):
            target.checkpoints.load_last(path,target)
        assert torch.equal(rng,torch.get_rng_state())
        for name,value in target.model.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        assert not target.optimizer.state
    finally:
        context.tracker.close()
        if target is not None:
            target.tracker.close()


def test_new_test_observations_with_same_input_semantics_do_not_require_training_source_versions(tmp_path):
    folder = od_fixture(tmp_path/'raw',features=1,grid_shape=(2,2))
    context = make_context(cstn_config(tmp_path,'saved'))
    try:
        path = context.checkpoints.save_best(context.model)
        source = folder/'ODTiny.gridod'
        frame = pd.read_csv(source)
        frame.loc[frame['time']==frame['time'].max(),'flow_0'] += 100
        frame.to_csv(source,index=False)
        cfg = context.config
        bundle = prepare_od(cfg['data'],cfg['training'],cfg['evaluation'],17)
        assert inference_spec(bundle,cfg)==inference_spec(context.bundle,cfg)
        CheckpointManager.load_weights(path,context.model,input_spec=inference_spec(bundle,cfg))
    finally:
        context.tracker.close()
