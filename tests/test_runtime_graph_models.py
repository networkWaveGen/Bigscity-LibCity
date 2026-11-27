import copy
import importlib

import numpy as np
import pytest
import torch

from libcity.executor.evaluate import model_inputs
from libcity.executor.train import TrainState, train_loop
from libcity.model.runtime_registry import model_spec, resolve
from libcity.utils.normalization import NoneScaler
from runtime_fixtures import atomic_fixture
from runtime_helpers import make_context, tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')

SMALL = {
    'STGCN': dict(Ks=2,Kt=2,blocks=[[1,4,4],[4,4,4]],dropout=0.),
    'DCRNN': dict(rnn_units=4,num_rnn_layers=1,max_diffusion_step=2,filter_type='dual_random_walk',
                  cl_decay_steps=1,use_curriculum_learning=True),
    'Trafformer': dict(d_model=8,n_heads=2,d_ff=16,e_layers=1,d_layers=1,dropout=0.),
}


def configuration(root,name,experiment='graph',epochs=1):
    config = tiny_config(root,experiment,epochs=epochs)
    config['model'] = dict(name=name,params=SMALL[name])
    config['data']['input_window'] = 12 if name=='STGCN' else 2
    config['training']['accumulation_steps'] = 1
    return config


@pytest.fixture
def data(tmp_path):
    atomic_fixture(tmp_path/'raw')
    return tmp_path


@pytest.mark.parametrize('name', ['STGCN','DCRNN','Trafformer'])
def test_forward_loss_and_short_updates_match_legacy_computation(data,name):
    context = make_context(configuration(data,name))
    metadata = context.bundle.metadata
    config = dict(context.config['model']['params'],input_window=context.config['data']['input_window'],
        output_window=2,device=torch.device('cpu'),output_dim=1,stgcn_train_mode='full',gradient_accumulation_steps=1)
    features = dict(scaler=NoneScaler(),num_nodes=metadata.num_nodes,feature_dim=metadata.feature_dim,
                    output_dim=1,adj_mx=metadata.adjacency.copy())
    module = importlib.import_module('libcity.model.traffic_speed_prediction.'+name)
    torch.manual_seed(31)
    legacy = getattr(module,name)(config,features)
    loaded = context.model.load_state_dict(legacy.state_dict(),strict=False)
    assert not loaded.unexpected_keys
    assert all('support_' in key or key.endswith('Lk') for key in loaded.missing_keys)
    batches = list(context.bundle.train)[:2]
    legacy.eval()
    context.model.eval()
    with torch.no_grad():
        expected = legacy.predict(batches[0])
        actual = context.model(model_inputs(batches[0],1))
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    legacy_optimizer = torch.optim.Adam(legacy.parameters(),lr=.001)
    context.optimizer = torch.optim.Adam(context.model.parameters(),lr=.001)
    spec = model_spec(name)
    step = resolve(spec.step)(context.loss_fn) if spec.step else None
    legacy.train()
    context.model.train()
    for index,batch in enumerate(batches):
        random_state = np.random.get_state()
        legacy_optimizer.zero_grad()
        expected = legacy.calculate_loss(batch,index) if name=='DCRNN' else legacy.calculate_loss(batch)
        expected.backward()
        legacy_optimizer.step()
        np.random.set_state(random_state)
        context.optimizer.zero_grad()
        state = TrainState(batches_seen=index)
        if step:
            result = step(context.model,batch,state)
        else:
            result = context.loss_fn(context.model(model_inputs(batch,1)),batch['y'])
        assert result.loss.item() == pytest.approx(expected.item(),rel=1e-5)
        result.loss.backward()
        context.optimizer.step()
    old_parameters = dict(legacy.named_parameters())
    for key,value in context.model.named_parameters():
        torch.testing.assert_close(value,old_parameters[key],rtol=1e-5,atol=1e-6)
    assert not any(hasattr(context.model,key) for key in ('_scaler','_logger','device'))
    if name=='STGCN':
        assert 'Lk' in dict(context.model.named_buffers())
        assert 'Lk' in dict(context.model.st_conv1.sconv.named_buffers())
        context.model.double().eval()
        assert context.model.st_conv1.sconv.Lk.dtype==torch.float64
        with torch.no_grad():
            assert torch.isfinite(context.model({'X':batches[0]['X'].double()})).all()
    context.tracker.close()


def test_dcrnn_curriculum_resume_retains_progress_rng_and_next_updates(data):
    cfg = configuration(data,'DCRNN','full',epochs=2)
    first = make_context(cfg)
    step = resolve(model_spec('DCRNN').step)(first.loss_fn)
    train_loop(first,step)
    expected = torch.load(first.directory/'checkpoints/last.pt',weights_only=True)
    half = make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'half'}})
    train_loop(half,resolve(model_spec('DCRNN').step)(half.loss_fn),until_epoch=1)
    resumed = make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'resumed'}})
    resumed.checkpoints.load_last(half.directory/'checkpoints/last.pt',resumed)
    assert resumed.state.batches_seen == len(half.bundle.train)
    train_loop(resumed,resolve(model_spec('DCRNN').step)(resumed.loss_fn))
    actual = torch.load(resumed.directory/'checkpoints/last.pt',weights_only=True)
    assert actual['train_state'] == expected['train_state']
    assert actual['train_state']['task_state']['curriculum_batches_seen'] == actual['train_state']['batches_seen']-1
    for key in expected['model']:
        if expected['model'][key].is_sparse:
            torch.testing.assert_close(actual['model'][key].to_dense(),expected['model'][key].to_dense(),rtol=0,atol=0)
        else:
            torch.testing.assert_close(actual['model'][key],expected['model'][key],rtol=0,atol=0)
    for context in (first,half,resumed):
        context.tracker.close()


def test_trafformer_unequal_horizon_flushes_short_accumulation_group(data):
    cfg = configuration(data,'Trafformer')
    cfg['data']['input_window'] = 4
    cfg['training']['accumulation_steps'] = 16
    cfg['training']['optimizer'] = {'name':'sgd','lr':.001}
    context = make_context(cfg)
    reference = copy.deepcopy(context.model)
    optimizer = torch.optim.SGD(reference.parameters(),lr=.001)
    batches = list(context.bundle.train)
    assert len(batches)<16
    combined = {key:torch.cat([batch[key] for batch in batches]) for key in batches[0]}
    reference.train()
    prediction = reference(model_inputs(combined,1))
    assert prediction.shape[1] == 2
    result = context.loss_fn(prediction,combined['y'])
    result.loss.backward()
    optimizer.step()
    train_loop(context)
    assert context.state.optimizer_steps == 1
    actual = torch.load(context.directory/'checkpoints/last.pt',weights_only=True)['model']
    for key,value in reference.state_dict().items():
        torch.testing.assert_close(actual[key],value,rtol=1e-4,atol=2e-6)
    context.tracker.close()


@pytest.mark.parametrize('name',['STGCN','DCRNN','Trafformer'])
def test_graph_models_use_shared_experiment_entry_and_inference_weights(data,name):
    from libcity.pipeline.experiment import run_experiment
    cfg = configuration(data,name)
    trained = run_experiment('train',config=cfg)
    evaluated = run_experiment('evaluate',config=cfg,checkpoint=trained.artifacts['best'],
        overrides={'runtime':{'exp_id':'independent'}})
    assert np.isfinite(list(trained.metrics.values())).all()
    for key,value in evaluated.metrics.items():
        assert value==pytest.approx(trained.metrics[key])
