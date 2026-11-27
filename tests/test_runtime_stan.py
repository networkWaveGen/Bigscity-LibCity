"""Causal two-level attention, exact static diameter and finite sampled softmax."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config
from libcity.data.stan_trajectory import spherical_diameter,spherical_distance
from libcity.data.dataset.trajectory_encoder.stan_encoder import haversine
from libcity.executor.stan_step import sampled_step
from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.STAN import STAN as ReferenceSTAN
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import strnn_fixture,task_fixture
from runtime_helpers import make_context


def configuration(root,experiment):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='STAN',params=dict(embed_dim=4,max_len=8,candidate_chunk_size=3)),
        training=dict(epochs=3,batch_size=8,shuffle=True,loss=dict(negative_samples=3),accumulation_steps=2,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


@pytest.mark.parametrize('coordinates',[
    [[116,39],[116.001,39.002],[116.02,39.01],[116.004,39.02]],
    [[0,0],[180,0],[179.9,89.9],[-179.9,-89.9],[0,90],[0,-90]],
    [[116,39],[116,39]],
    (np.array([116.,39.])+np.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]])*1e-8).tolist(),
])
def test_static_spherical_diameter_matches_all_pairs_without_storing_pair_table(coordinates):
    coordinates = np.asarray(coordinates,dtype=np.float64)
    expected = max(haversine(*left,*right) for left in coordinates for right in coordinates)
    assert spherical_diameter(coordinates)==pytest.approx(expected,rel=1e-10,abs=1e-12)
    rng = np.random.default_rng(17)
    global_points = np.column_stack((rng.uniform(-180,180,30),rng.uniform(-89,89,30)))
    expected = max(haversine(*left,*right) for left in global_points for right in global_points)
    assert spherical_diameter(global_points)==pytest.approx(expected,abs=1e-6)


def test_original_two_level_scores_and_all_gradients_with_explicit_mask_fix(tmp_path):
    strnn_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'reference')
    config['model']['params']['time_upper'] = 2.
    context = make_context(config)
    try:
        model,info = context.model,context.bundle.metadata.info
        coordinates = info['coordinates']+info['coordinate_origin']
        spatial = spherical_distance(coordinates[:,None],coordinates[None,:])
        reference = ReferenceSTAN(dict(max_session_len=8,embed_dim=4,batch_size=2,device=torch.device('cpu'),num_neg=3),
            dict(tim_size=169,loc_size=9,uid_size=3,ex=[model.spatial_upper,0.,model.temporal_upper,0.],spatial_matrix=spatial.tolist()))
        mapping = {name:getattr(reference.MultiEmbed,name).weight for name in
            ('emb_t','emb_l','emb_u','emb_su','emb_sl','emb_tu','emb_tl')}
        mapping.update({name:getattr(reference.SelfAttn,name).weight for name in ('query','key','value')})
        mapping['position'] = reference.Attn.value.weight
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                parameter.copy_(mapping[name.removesuffix('.weight')])
        # The old post-softmax mask is deliberately corrected in the reference.
        def attention(joint,delta,lengths):
            valid = torch.arange(joint.shape[1])[None]<lengths[:,None]
            logits = reference.SelfAttn.query(joint)@reference.SelfAttn.key(joint).transpose(-1,-2)+delta.sum(-1)
            weights = logits.masked_fill(~valid[:,None,:],-torch.inf).softmax(-1).masked_fill(~valid[:,:,None],0.)
            return weights@reference.SelfAttn.value(joint)
        reference.SelfAttn.forward = attention
        batch = dict(current_loc=torch.tensor([[1,2,3],[3,4,0]]),current_tim=torch.tensor([[1,2,3],[4,5,0]]),
            lengths=torch.tensor([3,2]),uid=torch.tensor([1,2]),elapsed_hours=torch.tensor([[0.,.1,.25],[0.,.2,0.]]),
            query_offset_hours=torch.tensor([.05,.1]))
        traj = torch.zeros(2,8,3,dtype=torch.long)
        pair = torch.zeros(2,8,8,2)
        temporal = torch.zeros(2,8)
        for index,length in enumerate(batch['lengths']):
            count = int(length)
            locations = batch['current_loc'][index,:count]
            elapsed = batch['elapsed_hours'][index,:count]
            traj[index,:count,0] = batch['uid'][index]
            traj[index,:count,1] = locations
            traj[index,:count,2] = batch['current_tim'][index,:count]
            pair[index,:count,:count,0] = reference.mat2s[locations[:,None]-1,locations[None,:]-1]
            pair[index,:count,:count,1] = (elapsed[:,None]-elapsed[None,:]).abs()
            temporal[index,:count] = elapsed[-1]+batch['query_offset_hours'][index]-elapsed
        old = reference(traj,pair,temporal,batch['lengths'])
        actual = model(batch)
        torch.testing.assert_close(actual,F.log_softmax(old,dim=-1),rtol=2e-5,atol=2e-6)
        candidates = torch.tensor([0,3,5,7])
        selected = model.forward_candidates(batch,candidates)
        torch.testing.assert_close(selected,old[:,candidates],rtol=2e-5,atol=2e-6)
        labels = torch.tensor([0,1])
        F.cross_entropy(selected,labels).backward()
        F.cross_entropy(old[:,candidates],labels).backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,mapping[name.removesuffix('.weight')].grad,rtol=3e-4,atol=2e-6)
        assert not hasattr(model,'mat2s') and model.coordinates.shape==(8,2)
        assert all(not hasattr(module,'device') and not hasattr(module,'_scaler') for module in model.modules())
    finally:
        context.tracker.close()


def test_complete_sampled_training_full_ranking_and_exact_rng_resume(tmp_path):
    strnn_fixture(tmp_path/'raw')
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,sampled_step(partial.loss_fn),until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state'] and expected['scheduler']==actual['scheduler']
    torch.testing.assert_close(expected['rng']['torch'],actual['rng']['torch'],rtol=0,atol=0)
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
    records = json.loads(evaluated.artifacts['predictions'].read_text())
    assert records and all(len(row['prediction'])==8 and row['truth'] in range(10,18) for row in records)
    info = json.loads(evaluated.artifacts['runtime_info'].read_text())
    assert info['benchmark_protocol']=='stan_ranking'


def test_masked_prefix_batch_chunk_and_padding_independence_float64(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'padding'))
    try:
        model = context.model.double().eval()
        batch = next(iter(context.bundle.train))
        inputs = context.loss_fn.inputs(batch)
        before = {key:value.clone() for key,value in inputs.items()}
        prediction = model(inputs)
        for index in range(len(batch['y'])):
            one = {key:value[index:index+1] for key,value in inputs.items()}
            torch.testing.assert_close(model(one),prediction[index:index+1],rtol=1e-12,atol=1e-12)
        valid = torch.arange(inputs['current_loc'].shape[1])[None]<inputs['lengths'][:,None]
        poisoned = {key:value.clone() for key,value in inputs.items()}
        for key in ('current_loc','current_tim'):
            poisoned[key][~valid] = 999999
        poisoned['elapsed_hours'][~valid] = float('nan')
        poisoned.update(y=torch.full_like(batch['y'],9999),target_tim=torch.tensor(float('nan')))
        rng = torch.get_rng_state().clone()
        torch.testing.assert_close(model(poisoned),prediction,rtol=0,atol=0)
        model.chunk_size = 1
        torch.testing.assert_close(model(inputs),prediction,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        assert prediction.dtype==torch.float64 and prediction.shape==(len(batch['y']),8)
        torch.testing.assert_close(prediction.exp().sum(-1),torch.ones(len(prediction),dtype=torch.float64))
        for key,value in before.items():
            torch.testing.assert_close(inputs[key],value,rtol=0,atol=0)
        context.loss_fn(prediction,batch['y']).loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        with pytest.raises(ValueError,match='query_offset_hours'):
            model({**inputs,'query_offset_hours':torch.full((len(batch['y']),),-1.)})
        model.max_len = 1
        with pytest.raises(ValueError,match='max_len'):
            model(inputs)
    finally:
        context.tracker.close()


def test_hour_of_week_users_train_bounds_and_target_time_position_isolation(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'features'))
    try:
        frame = pd.read_csv(folder/'TaskTiny.dyna')
        frame['time'] = pd.to_datetime(frame.time,utc=True)
        groups = {identity:part.sort_values('time') for identity,part in frame.groupby('traj_id')}
        bundle = context.bundle
        assert bundle.metadata.info['uid_size']==3 and bundle.metadata.info['tim_pad']==0
        assert bundle.metadata.input_layout['time_encoding']=='hour_of_week_1_to_168'
        assert bundle.metadata.info['temporal_upper']==max(float(row['elapsed_hours'][-1]) for row in bundle.train.dataset.rows)
        for loader in (bundle.train,bundle.val,bundle.test):
            for row in loader.dataset.rows:
                target = frame[frame.dyna_id==int(row['sample_id'])].iloc[0]
                prefix = groups[target.traj_id].iloc[:int(row['lengths'])]
                expected = torch.tensor((prefix.time.dt.hour+24*prefix.time.dt.day_of_week+1).to_numpy(),dtype=torch.long)
                elapsed = torch.tensor((prefix.time-prefix.time.iloc[0]).dt.total_seconds().to_numpy()/3600,dtype=torch.float32)
                torch.testing.assert_close(row['current_tim'],expected,rtol=0,atol=0)
                torch.testing.assert_close(row['elapsed_hours'],elapsed,rtol=0,atol=0)
                assert int(row['uid']) in (1,2)
        sample = bundle.test.dataset.rows[-1]
        identity = int(sample['sample_id'])
        batch = bundle.test.collate_fn([sample])
        prediction = context.model.eval()(context.loss_fn.inputs(batch))
        state = {key:value.clone() for key,value in context.model.state_dict().items()}
        temporal_upper = bundle.metadata.info['temporal_upper']
    finally:
        context.tracker.close()
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    index = frame.index[frame.dyna_id==identity][0]
    frame.loc[index,'time'] = (pd.Timestamp(frame.loc[index,'time'])+pd.Timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.loc[index,'location'] = 10+(int(frame.loc[index,'location'])-10+3)%8
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    changed = make_context(configuration(tmp_path,'changed-target'))
    try:
        assert changed.bundle.metadata.info['temporal_upper']==temporal_upper
        changed.model.eval().load_state_dict(state)
        sample = next(row for row in changed.bundle.test.dataset.rows if int(row['sample_id'])==identity)
        after = changed.bundle.test.collate_fn([sample])
        for key,value in changed.loss_fn.inputs(batch).items():
            torch.testing.assert_close(after[key],value,rtol=0,atol=0)
        torch.testing.assert_close(changed.model(changed.loss_fn.inputs(after)),prediction,rtol=0,atol=0)
    finally:
        changed.tracker.close()


def test_unique_positive_sampler_handles_duplicates_all_classes_and_first_class_negatives(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'sampling'))
    try:
        loss = context.loss_fn
        loss.negative_samples = 100
        target = torch.tensor([0,0,2,2])
        candidates,labels = loss.candidates(target)
        assert len(candidates)==8 and len(set(candidates.tolist()))==8
        torch.testing.assert_close(labels,torch.tensor([0,0,1,1]))
        torch.testing.assert_close(candidates[labels],target)
        assert set(candidates[2:].tolist())=={1,3,4,5,6,7}
        rng = torch.get_rng_state().clone()
        candidates,labels = loss.candidates(torch.arange(8))
        torch.testing.assert_close(candidates,torch.arange(8))
        torch.testing.assert_close(labels,torch.arange(8))
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        candidates,labels = loss.candidates(torch.tensor([7]))
        assert 0 in candidates[1:] and int(candidates[labels[0]])==7
    finally:
        context.tracker.close()


def test_sampled_step_matches_selected_full_logits_and_gradients_without_full_forward(tmp_path,monkeypatch):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'selected'))
    try:
        batch = next(iter(context.bundle.train))
        loss,model = context.loss_fn,context.model
        rng = torch.get_rng_state().clone()
        candidates,labels = loss.candidates(batch['y'])
        full = model.score(*model.encode(loss.inputs(batch)),torch.arange(loss.output_dim))
        expected = F.cross_entropy(full[:,candidates],labels)
        expected.backward()
        gradients = {name:parameter.grad.clone() for name,parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        def forbidden(*args,**kwargs):
            raise AssertionError('sampled training must not call full-vocabulary forward')
        monkeypatch.setattr(model,'forward',forbidden)
        actual = sampled_step(loss)(model,batch,context.state)
        torch.testing.assert_close(actual.loss,expected,rtol=1e-5,atol=1e-6)
        actual.loss.backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,gradients[name],rtol=2e-4,atol=1e-6)
    finally:
        context.tracker.close()


def test_zero_time_coincident_points_and_single_location_are_finite(tmp_path):
    folder = task_fixture(tmp_path/'raw')
    geo = pd.read_csv(folder/'TaskTiny.geo').iloc[:1]
    geo.to_csv(folder/'TaskTiny.geo',index=False)
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    frame['location'] = 10
    frame['time'] = frame.groupby('traj_id')['time'].transform('first')
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    config = configuration(tmp_path,'degenerate')
    config['training']['epochs'] = 1
    result = run_experiment('train',config=config)
    assert result.metrics['loss']==0. and result.metrics['Recall@1']==1.
    info = json.loads(result.artifacts['runtime_info'].read_text())
    assert info['split']['temporal_upper']==info['split']['spatial_upper']==1.


def test_protocol_rejects_wrong_loss_and_nonpositive_negative_request(tmp_path):
    config = configuration(tmp_path,'protocol')
    assert load_config(config=config)['training']['loss']['name']=='sampled_softmax'
    config['training']['loss'] = dict(name='nll')
    with pytest.raises(ValueError,match='training.loss'):
        load_config(config=config)
    config['training']['loss'] = dict(negative_samples=0)
    with pytest.raises(ValueError,match='negative_samples'):
        load_config(config=config)


def test_insufficient_position_capacity_fails_during_construction(tmp_path):
    strnn_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'too-short')
    config['model']['params']['max_len'] = 1
    with pytest.raises(ValueError,match='max_len must cover'):
        make_context(config)


def test_invalid_static_coordinates_fail_at_data_entry(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    geo = pd.read_csv(folder/'TaskTiny.geo')
    geo.loc[0,'coordinates'] = '[116, 100]'
    geo.to_csv(folder/'TaskTiny.geo',index=False)
    with pytest.raises(ValueError,match='point coordinates'):
        make_context(configuration(tmp_path,'invalid'))


def test_actual_cpu_amp_training(tmp_path):
    strnn_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0
