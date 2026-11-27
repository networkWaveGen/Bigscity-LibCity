import copy
import importlib
import json

import numpy as np
import pytest
import torch

from libcity.executor.astgnn_train import train as train_astgnn,prepare_resume
from libcity.executor.mtgnn_train import train as train_mtgnn,subgraphs
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


@pytest.fixture
def data(tmp_path):
    atomic_fixture(tmp_path/'raw',interval_minutes=30)
    return tmp_path


def ast_config(root,name='ast',TE=True):
    config = tiny_config(root,name,epochs=2,scheduler={'name':'step','interval':'epoch','step_size':1,'gamma':.7})
    config['model'] = dict(name='ASTGNN',params=dict(num_layers=1,d_model=8,nb_head=2,TE=TE,dropout=0.))
    config['training']['strategy'] = dict(fine_tune_epochs=2,fine_tune_lr=.0003)
    return config


def test_astgnn_teacher_labels_and_original_scale_short_update_match(data):
    settings = ast_config(data,TE=False)
    settings['data']['scaler']={'name':'standard'}
    context = make_context(settings)
    metadata = context.bundle.metadata
    assert metadata.input_layout['kind']=='cpt'
    assert context.bundle.split_summary['input_window']==2
    config = dict(context.config['model']['params'],device=torch.device('cpu'),output_window=2,
                  points_per_hour=2,len_closeness=1,len_period=0,len_trend=0)
    old_module = importlib.import_module('libcity.model.traffic_flow_prediction.ASTGNN')
    old = old_module.ASTGNN(config,dict(feature_dim=metadata.feature_dim,output_dim=1,num_nodes=metadata.num_nodes,
                                        adj_mx=metadata.adjacency.copy(),scaler=context.bundle.scaler))
    loaded = context.model.load_state_dict(old.state_dict(),strict=False)
    assert not loaded.unexpected_keys
    assert all(key.endswith('sym_norm_Adj_matrix') for key in loaded.missing_keys)
    old.eval()
    context.model.eval()
    batches = list(context.bundle.train)[:2]
    with torch.no_grad():
        torch.testing.assert_close(context.model({'X':batches[0]['X']}),old.predict(batches[0]),rtol=1e-5,atol=1e-6)
    optimizer = torch.optim.Adam(old.parameters(),lr=.001)
    old.train()
    context.model.train()
    for batch in batches:
        optimizer.zero_grad()
        expected = old.calculate_train_loss(batch)
        expected.backward()
        optimizer.step()
        context.optimizer.zero_grad()
        prediction = context.model.forward_with_teacher({'X':batch['X']},batch['y'])
        actual = context.loss_fn(prediction,batch['y'])
        assert actual.loss.item()==pytest.approx(expected.item(),rel=1e-5)
        actual.loss.backward()
        context.optimizer.step()
    parameters = dict(old.named_parameters())
    for key,value in context.model.named_parameters():
        torch.testing.assert_close(value,parameters[key],rtol=1e-5,atol=1e-6)
    context.tracker.close()


@pytest.mark.parametrize('interrupted_stage',['teacher','fine_tune'])
def test_astgnn_multiple_stages_resume_and_reset_optimizer_once(data,interrupted_stage):
    cfg = ast_config(data)
    full = make_context(cfg)
    train_astgnn(full)
    expected = torch.load(full.directory/'checkpoints/last.pt',weights_only=True)
    partial = make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'partial'}})
    if interrupted_stage=='teacher':
        train_astgnn(partial,until_teacher_epoch=1)
    else:
        train_astgnn(partial,until_fine_epoch=1)
    resumed = make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'resumed'}})
    resumed.checkpoints.load_last(partial.directory/'checkpoints/last.pt',resumed,prepare=prepare_resume)
    train_astgnn(resumed)
    actual = torch.load(resumed.directory/'checkpoints/last.pt',weights_only=True)
    assert actual['train_state']==expected['train_state']
    assert actual['train_state']['stage']=='fine_tune'
    assert actual['train_state']['task_state']['optimizer_resets']==1
    assert actual['train_state']['optimizer_steps']==4*len(full.bundle.train)
    assert next(iter(actual['optimizer']['state'].values()))['step'].item()==2*len(full.bundle.train)
    assert actual['optimizer']['param_groups'][0]['lr']==pytest.approx(.0003*.7**2)
    assert actual['scheduler']==expected['scheduler']
    for key in actual['model']:
        torch.testing.assert_close(actual['model'][key],expected['model'][key],rtol=0,atol=0)
    rows = [json.loads(line) for line in (full.directory/'metrics.jsonl').read_text().splitlines()]
    assert [row['stage'] for row in rows]==['teacher','teacher','fine_tune','fine_tune']
    for context in (full,partial,resumed):
        context.tracker.close()


def test_astgnn_temporal_encoding_keeps_projection_gradients_and_pipeline(data):
    cfg = ast_config(data)
    context = make_context(cfg)
    batch = next(iter(context.bundle.train))
    prediction = context.model.forward_with_teacher({'X':batch['X']},batch['y'])
    value = context.loss_fn(prediction,batch['y'])
    value.loss.backward()
    gradient = context.model.src_embed[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    context.tracker.close()
    cfg['runtime']['exp_id']='pipeline'
    result = run_experiment('train',config=cfg)
    info = json.loads(result.artifacts['runtime_info'].read_text())
    assert info['protocol_changes']['data.effective_input_window']==2
    evaluated = run_experiment('evaluate',config=cfg,checkpoint=result.artifacts['best'],
        overrides={'runtime':{'exp_id':'independent'}})
    assert evaluated.metrics['loss']==pytest.approx(result.metrics['loss'])


def mtg_config(root,name='mtg',epochs=1):
    config = tiny_config(root,name,epochs=epochs)
    config['model'] = dict(name='MTGNN',params=dict(dropout=0.,subgraph_size=3,node_dim=4,
        conv_channels=4,residual_channels=4,skip_channels=4,end_channels=8,layers=1))
    config['training']['strategy'] = dict(num_split=2,permutation_interval=100,curriculum_step=2,use_curriculum_learning=True)
    return config


def old_mtg(context):
    module = importlib.import_module('libcity.model.traffic_speed_prediction.MTGNN')
    metadata = context.bundle.metadata
    config = dict(context.config['model']['params'],input_window=context.config['data']['input_window'],output_window=2,
                  output_dim=1,device=torch.device('cpu'),use_curriculum_learning=True,step_size1=2)
    old = module.MTGNN(config,dict(num_nodes=metadata.num_nodes,feature_dim=metadata.feature_dim,
        scaler=context.bundle.scaler,adj_mx=metadata.adjacency.copy(),num_batches=len(context.bundle.train)))
    loaded = context.model.load_state_dict(old.state_dict(),strict=False)
    assert not loaded.unexpected_keys
    assert set(loaded.missing_keys)=={'idx','predefined_A'}
    return old


def test_mtg_model_matches_legacy_forward_loss_and_adam_updates(data):
    context = make_context(mtg_config(data))
    old = old_mtg(context)
    old.eval()
    context.model.eval()
    batches=list(context.bundle.train)[:2]
    with torch.no_grad():
        torch.testing.assert_close(context.model({'X':batches[0]['X']}),old.predict(batches[0]),rtol=1e-5,atol=1e-6)
    optimizer=torch.optim.Adam(old.parameters(),lr=.001)
    old.train()
    context.model.train()
    for index,batch in enumerate(batches):
        optimizer.zero_grad()
        loss=old.calculate_loss(batch,batches_seen=index)
        loss.backward()
        optimizer.step()
        context.optimizer.zero_grad()
        prediction=context.model({'X':batch['X']})
        level=min(2,1+index//2)
        value=context.loss_fn(prediction[:,:level],batch['y'][:,:level])
        assert value.loss.item()==pytest.approx(loss.item(),rel=1e-5)
        value.loss.backward()
        context.optimizer.step()
    for key,value in context.model.named_parameters():
        torch.testing.assert_close(value,dict(old.named_parameters())[key],rtol=1e-5,atol=1e-6)
    context.tracker.close()


def test_mtg_multiple_updates_match_corrected_reference_and_lr(data):
    cfg=mtg_config(data)
    cfg['training']['optimizer']={'name':'sgd','lr':.001}
    cfg['training']['scheduler']={'name':'step','interval':'step','step_size':3,'gamma':.9}
    context=make_context(cfg)
    old=old_mtg(context).train()
    optimizer=torch.optim.SGD(old.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.9)
    random_state=np.random.get_state()
    progress=0
    for index,batch in enumerate(context.bundle.train):
        if index%100==0:
            permutation=np.random.permutation(context.bundle.metadata.num_nodes)
        for nodes in subgraphs(permutation,2):
            # Reference fixes the old executor's stale-gradient carry between
            # optimizer steps, and bounds top-k to the current node subset.
            old.gc.k=min(3,len(nodes))
            optimizer.zero_grad()
            loss=old.calculate_loss(batch,idx=nodes,batches_seen=progress)
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress+=1
    np.random.set_state(random_state)
    train_mtgnn(context)
    state=torch.load(context.directory/'checkpoints/last.pt',weights_only=True)
    assert context.state.optimizer_steps==progress==2*len(context.bundle.train)
    assert context.state.batches_seen==len(context.bundle.train)
    assert state['optimizer']['param_groups'][0]['lr']==pytest.approx(optimizer.param_groups[0]['lr'])
    for key,value in old.named_parameters():
        torch.testing.assert_close(state['model'][key],value,rtol=1e-5,atol=1e-6)
    context.tracker.close()


def test_mtg_curriculum_subgraphs_and_step_schedule_resume_exactly(data):
    cfg=mtg_config(data,epochs=2)
    cfg['training']['scheduler']={'name':'onecycle','interval':'step','max_lr':.002}
    full=make_context(cfg)
    train_mtgnn(full)
    expected=torch.load(full.directory/'checkpoints/last.pt',weights_only=True)
    partial=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'partial'}})
    train_mtgnn(partial,until_epoch=1)
    resumed=make_context({**cfg,'runtime':{**cfg['runtime'],'exp_id':'resumed'}})
    resumed.checkpoints.load_last(partial.directory/'checkpoints/last.pt',resumed)
    train_mtgnn(resumed)
    actual=torch.load(resumed.directory/'checkpoints/last.pt',weights_only=True)
    assert actual['train_state']==expected['train_state']
    assert actual['scheduler']==expected['scheduler']
    assert actual['train_state']['task_state']['subgraph_steps']==4*len(full.bundle.train)
    for key in actual['model']:
        torch.testing.assert_close(actual['model'][key],expected['model'][key],rtol=0,atol=0)
    for context in (full,partial,resumed):
        context.tracker.close()


def test_mtg_independent_trainer_uses_pipeline_and_prediction_switch(data):
    cfg=mtg_config(data)
    cfg['evaluation']={'save_predictions':False}
    result=run_experiment('train',config=cfg)
    assert 'predictions' not in result.artifacts
    assert not (data/'runs/mtg/eval/predictions.npz').exists()
    evaluated=run_experiment('evaluate',config=cfg,checkpoint=result.artifacts['best'],
                            overrides={'runtime':{'exp_id':'independent'}})
    assert evaluated.metrics['loss']==pytest.approx(result.metrics['loss'])


def test_mtg_static_graph_keeps_self_support_for_single_node_subgraphs(data):
    cfg=mtg_config(data)
    cfg['model']['params']['buildA_true']=False
    result=run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()
    saved=torch.load(result.artifacts['last'],weights_only=True)
    assert saved['train_state']['optimizer_steps']==2*saved['train_state']['batches_seen']
    assert torch.all(saved['model']['predefined_A'].diag()==0)
