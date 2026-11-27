"""CPT branch identity, reference math and complete multiscale graph lifecycles."""
import importlib
import json

import pandas as pd
import pytest
import torch

from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,name,experiment):
    config = tiny_config(root,experiment,epochs=3,batch_size=64,shuffle=True)
    config['data'].update(output_window=1,sampling=dict(name='cpt',closeness=2,period=1,trend=1))
    config['model'] = dict(name=name,params=dict(nb_block=2,K=2,nb_chev_filter=4,nb_time_filter=4))
    return config


def fixture(root):
    return atomic_fixture(root/'raw',steps=400,interval_minutes=60)


@pytest.mark.parametrize('name',['ASTGCN','MSTGCN'])
def test_reference_forward_gradients_and_explicit_cpt_branch_order(tmp_path,name):
    fixture(tmp_path)
    context = make_context(configuration(tmp_path,name,'reference'))
    try:
        model = context.model.eval()
        metadata = context.bundle.metadata
        legacy_type = getattr(importlib.import_module('libcity.model.traffic_flow_prediction.'+name),name)
        reference = legacy_type(dict(context.config['model']['params'],output_window=1,device=torch.device('cpu')),
            dict(num_nodes=3,feature_dim=1,output_dim=1,adj_mx=metadata.adjacency.copy(),
                 len_closeness=2,len_period=1,len_trend=1,scaler=NoneScaler())).eval()
        loaded = model.load_state_dict(reference.state_dict(),strict=False)
        assert not loaded.unexpected_keys
        assert loaded.missing_keys and all(key.endswith('.basis') for key in loaded.missing_keys)
        batch = next(iter(context.bundle.train))
        pieces = {attribute.split('_')[0]:batch['X'][:,start:end] for attribute,start,end in model.segments}
        assert list(pieces)==['weeks','days','hours']
        legacy_source = torch.cat([pieces['hours'],pieces['days'],pieces['weeks']],dim=1)
        actual = model(model_inputs(batch,1))
        expected = reference({'X':legacy_source})
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for key,value in model.named_parameters():
            torch.testing.assert_close(value.grad,dict(reference.named_parameters())[key].grad,rtol=1e-4,atol=1e-5)
        assert all(not any(hasattr(module,key) for key in ('device','DEVICE','_logger','_scaler')) for module in model.modules())
        assert any(key.endswith('basis') for key in dict(model.named_buffers()))
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name',['ASTGCN','MSTGCN'])
def test_all_cpt_branches_train_evaluate_and_resume_exactly(tmp_path,name):
    fixture(tmp_path)
    full = run_experiment('train',config=configuration(tmp_path,name,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,name,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    info = json.loads(full.artifacts['runtime_info'].read_text())
    assert info['protocol_changes']['data.sampling']['name']=='cpt'
    assert info['split']['history_span_points']==168
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,name,'partial'))
    train_loop(partial,until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,name,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state']
    assert expected['scheduler']==actual['scheduler']
    for key,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)


@pytest.mark.parametrize('name',['ASTGCN','MSTGCN'])
def test_unequal_filters_multiple_targets_batch_one_float64_and_no_labels(tmp_path,name):
    folder = atomic_fixture(tmp_path/'raw',steps=100)
    frame = pd.read_csv(folder/'Tiny.dyna')
    frame['flow'] = frame['speed']*.5
    frame.to_csv(folder/'Tiny.dyna',index=False)
    metadata = json.loads((folder/'config.json').read_text())
    metadata.update(data_col=['speed','flow'],output_dim=2)
    (folder/'config.json').write_text(json.dumps(metadata))
    config = configuration(tmp_path,name,'unequal')
    config['data'].update(output_window=2,sampling=dict(name='cpt',closeness=2,period=0,trend=0))
    config['model']['params']['nb_time_filter'] = 6
    context = make_context(config)
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        inputs = {key:value[:1].double() for key,value in model_inputs(batch,2).items()}
        original = inputs['X'].clone()
        rng = torch.get_rng_state()
        prediction = model(inputs)
        assert prediction.shape==(1,2,3,2) and prediction.dtype==torch.float64
        torch.testing.assert_close(prediction,model({**inputs,'y':torch.full_like(prediction,float('nan'))}))
        torch.testing.assert_close(inputs['X'],original,rtol=0,atol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        prediction.square().mean().backward()
        assert all(value.grad is None or torch.isfinite(value.grad).all() for value in model.parameters())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name',['ASTGCN','MSTGCN'])
def test_cpu_amp_and_single_node_zero_laplacian(tmp_path,name):
    atomic_fixture(tmp_path/'raw',steps=80,nodes=1)
    config = configuration(tmp_path,name,'amp-single')
    config['data'].update(output_window=2,sampling=dict(name='cpt',closeness=2,period=0,trend=0))
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=config)
    assert result.metrics['loss']>=0
    state = torch.load(result.artifacts['best'],weights_only=True)
    assert all(torch.isfinite(value).all() for value in state.values())
