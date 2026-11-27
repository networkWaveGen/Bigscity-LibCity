"""Original pairwise scorer, causal full-vocabulary geometry and sampled resume."""
from dataclasses import replace
import json

from geopy.distance import geodesic
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config
from libcity.data.atst_trajectory import CandidateCollator
from libcity.data.geodesic import distances_from
from libcity.executor.atst_step import ATSTRankingLoss,pairwise_step
from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.ATSTLSTM import ATSTLSTM as ReferenceATSTLSTM
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import strnn_fixture
from runtime_helpers import make_context


def configuration(root,experiment):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='ATSTLSTM',params=dict(hidden_size=4)),
        training=dict(epochs=3,batch_size=8,shuffle=True,loss=dict(negative_samples=3),accumulation_steps=2,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


def test_wgs84_vectorized_distances_match_karney_including_antipodes_and_poles(monkeypatch):
    import libcity.data.geodesic as module
    rng = np.random.default_rng(17)
    points = np.column_stack((rng.uniform(-180,180,400),rng.uniform(-89.9,89.9,400)))
    points = np.concatenate((points,[[0,0],[180,0],[179.999999,.000001],[180,90],[10,-90],[-180,0],[.0000001,0],[0,.0000001]]))
    called = []
    def fallback(*args):
        called.append(args)
        return geodesic(*args)
    monkeypatch.setattr(module,'geodesic',fallback)
    for origin in ([0,0],[116,39],[179.999,80],[0,90]):
        actual = distances_from(origin,points)
        expected = np.array([geodesic(origin[::-1],point[::-1]).km for point in points])
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-6)
    assert called and np.isfinite(actual).all()
    assert distances_from([180,0],np.array([[-180,0]]))[0]==0.


class ReferenceBatch(dict):
    def get_origin_len(self,key):
        return self['lengths'].tolist()


def test_original_valid_candidate_scores_pairwise_loss_and_all_gradients(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference'))
    try:
        model = context.model
        reference = ReferenceATSTLSTM(dict(hidden_size=4,device=torch.device('cpu')),context.bundle.metadata.info)
        model.load_state_dict(reference.state_dict())
        row = next(row for row in context.bundle.train.dataset.rows if int(row['lengths'])==3)
        batch = context.bundle.train.collate_fn([row])
        target = batch['y']
        negatives = torch.tensor([[value for value in range(8) if value!=int(target[0])]])
        order = torch.cat((target[:,None],negatives),dim=1)
        chosen_dis = batch['candidate_dis'].gather(1,order)
        legacy = ReferenceBatch(current_loc=batch['current_loc'],current_dis=batch['dis'],current_tim=batch['tim_interval'],
            lengths=batch['lengths'],uid=batch['uid'],target_loc=target+1,target_dis=chosen_dis[:,0],
            target_tim=torch.full((1,),123.),loc_neg=negatives+1,dis_neg=chosen_dis[:,1:],tim_neg=torch.full_like(negatives,123.,dtype=torch.float32))
        inputs = {**context.loss_fn.inputs(batch),'query_offset_seconds':torch.tensor([123.])}
        actual = model(inputs)
        expected = reference(legacy)
        torch.testing.assert_close(actual.gather(1,order),expected,rtol=1e-5,atol=1e-6)
        actual_loss = context.loss_fn.sampled(actual,target,negatives).loss
        expected_loss = reference.calculate_loss(legacy)
        torch.testing.assert_close(actual_loss,expected_loss,rtol=1e-5,atol=1e-6)
        actual_loss.backward();expected_loss.backward()
        parameters = dict(reference.named_parameters())
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,parameters[name].grad,rtol=2e-4,atol=1e-6)
    finally:
        context.tracker.close()


def test_sampled_training_full_ranking_and_exact_rng_resume(tmp_path):
    strnn_fixture(tmp_path/'raw')
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,pairwise_step(partial.loss_fn),until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state']
    assert expected['scheduler']==actual['scheduler']
    torch.testing.assert_close(expected['rng']['torch'],actual['rng']['torch'],rtol=0,atol=0)
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
    records = json.loads(evaluated.artifacts['predictions'].read_text())
    assert records and all(len(row['prediction'])==8 and row['truth'] in range(10,18) for row in records)
    info = json.loads(evaluated.artifacts['runtime_info'].read_text())
    assert info['benchmark_protocol']=='atst_ranking'


def test_masked_attention_batch_independence_float64_and_label_isolation(tmp_path):
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
        poisoned['current_loc'][~valid] = 999999
        poisoned['tim_interval'][~valid] = float('nan')
        poisoned['dis'][~valid] = -100.
        poisoned.update(y=torch.full_like(batch['y'],9999),target_loc=torch.tensor(-100),target_tim=torch.tensor(float('nan')))
        rng = torch.get_rng_state().clone()
        torch.testing.assert_close(model(poisoned),prediction,rtol=0,atol=0)
        assert prediction.dtype==torch.float64 and prediction.shape==(len(batch['y']),8)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key,value in before.items():
            torch.testing.assert_close(value,inputs[key],rtol=0,atol=0)
        assert all(not hasattr(module,'device') and not hasattr(module,'_scaler') for module in model.modules())
        context.loss_fn(prediction,batch['y']).loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        with pytest.raises(ValueError,match='query_offset_seconds'):
            model({**inputs,'query_offset_seconds':torch.full((len(batch['y']),),-1.)})
    finally:
        context.tracker.close()


def test_observed_prefix_geometry_and_popularity_do_not_use_next_target(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'features'))
    try:
        bundle = context.bundle
        frame = pd.read_csv(folder/'TaskTiny.dyna')
        frame['time'] = pd.to_datetime(frame.time,utc=True)
        ids = list(bundle.metadata.info['location_ids'])
        weights = frame[frame.time.dt.tz_localize(None)<pd.Timestamp(bundle.split_summary['train_end'])].location.value_counts()
        np.testing.assert_array_equal(bundle.metadata.info['negative_weights'],[weights.get(identity,0) for identity in ids])
        geo = pd.read_csv(folder/'TaskTiny.geo')
        coordinates = np.array([json.loads(value) for value in geo.coordinates])
        for loader in (bundle.train,bundle.val,bundle.test):
            for row in loader.dataset.rows:
                assert float(row['tim_interval'][0])==100. and float(row['dis'][0])==1.
                collated = loader.collate_fn([row])
                last = int(row['current_loc'][-1])-1
                expected = torch.tensor([geodesic(coordinates[last,::-1],point[::-1]).km for point in coordinates])
                torch.testing.assert_close(collated['candidate_dis'][0],expected,rtol=1e-6,atol=1e-7)
                assert float(collated['candidate_dis'][0,last])==0.
        sample = bundle.test.dataset.rows[-1]
        identity = int(sample['sample_id'])
        inputs = bundle.test.collate_fn([sample])
        prediction = context.model.eval()(context.loss_fn.inputs(inputs))
        state = {key:value.clone() for key,value in context.model.state_dict().items()}
        original_weights = bundle.metadata.info['negative_weights'].copy()
    finally:
        context.tracker.close()
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    index = frame.index[frame.dyna_id==identity][0]
    frame.loc[index,'time'] = (pd.Timestamp(frame.loc[index,'time'])+pd.Timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.loc[index,'location'] = ids[(ids.index(int(frame.loc[index,'location']))+3)%8]
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    changed = make_context(configuration(tmp_path,'changed-target'))
    try:
        np.testing.assert_array_equal(changed.bundle.metadata.info['negative_weights'],original_weights)
        changed.model.eval().load_state_dict(state)
        sample = next(row for row in changed.bundle.test.dataset.rows if int(row['sample_id'])==identity)
        batch = changed.bundle.test.collate_fn([sample])
        for key,value in changed.loss_fn.inputs(inputs).items():
            torch.testing.assert_close(batch[key],value,rtol=0,atol=0)
        torch.testing.assert_close(changed.model(changed.loss_fn.inputs(batch)),prediction,rtol=0,atol=0)
    finally:
        changed.tracker.close()


def test_popularity_negatives_exclude_positive_and_loss_is_deterministic(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'sampling'))
    try:
        weights = np.array([9.,0.,1.,0.,0.,0.,0.,0.])
        metadata = replace(context.bundle.metadata,info={**context.bundle.metadata.info,'negative_weights':weights})
        loss = ATSTRankingLoss(replace(context.bundle,metadata=metadata),dict(negative_samples=10))
        assert loss.negative_samples==1
        target = torch.zeros(64,dtype=torch.long)
        assert (loss.sample(target,torch.device('cpu'))==2).all()
        regular = context.loss_fn
        target = torch.arange(8)
        negatives = regular.sample(target,torch.device('cpu'))
        assert negatives.shape==(8,3) and (negatives!=target[:,None]).all()
        assert all(len(set(row.tolist()))==3 for row in negatives)
        scores = torch.tensor([[2.,-1.,4.,0.,3.,1.,-2.,0.]]).expand(8,-1).clone().requires_grad_()
        normalized = F.normalize(scores,dim=-1)
        expected = F.softplus(normalized-normalized.gather(1,target[:,None])).scatter(1,target[:,None],0.).sum(-1).mean()
        rng = torch.get_rng_state().clone()
        actual = regular(scores,target).loss
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        actual.backward()
        assert torch.isfinite(scores.grad).all()
        zero = torch.zeros(2,8,requires_grad=True)
        regular(zero,torch.tensor([0,1])).loss.backward()
        assert torch.isfinite(zero.grad).all()
    finally:
        context.tracker.close()


def test_candidate_geometry_cache_is_bounded_owned_and_reused(monkeypatch):
    import libcity.data.atst_trajectory as module
    coordinates = np.column_stack((np.linspace(116,116.1,40),np.linspace(39,39.1,40)))
    collator = CandidateCollator(coordinates)
    rows = [dict(current_loc=torch.tensor([index+1]),lengths=torch.tensor(1)) for index in range(20)]
    result = collator(rows)
    assert len(collator.cache)==16 and result['candidate_dis'].shape==(20,40)
    assert len(collator.coordinates)==40 and not collator.coordinates.flags.writeable
    last = result['candidate_dis'][-1].clone()
    result['candidate_dis'][-1].add_(1000)
    def unexpected(*args):
        raise AssertionError('cached last-position geometry must not be recalculated')
    monkeypatch.setattr(module,'distances_from',unexpected)
    torch.testing.assert_close(collator(rows[-1:])['candidate_dis'][0],last,rtol=0,atol=0)
    coordinates[:] = 0.
    assert float(collator.coordinates[0,0])==116.


def test_training_selected_candidates_match_full_scores_and_gradients_without_full_forward(tmp_path,monkeypatch):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'selected'))
    try:
        batch = next(iter(context.bundle.train))
        model,loss = context.model,context.loss_fn
        rng = torch.get_rng_state().clone()
        negatives = loss.sample(batch['y'],torch.device('cpu'))
        expected = loss.sampled(model(loss.inputs(batch)),batch['y'],negatives)
        expected.loss.backward()
        gradients = {name:parameter.grad.clone() for name,parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        def forbidden(*args,**kwargs):
            raise AssertionError('sampled training must not score the full vocabulary')
        monkeypatch.setattr(model,'forward',forbidden)
        actual = pairwise_step(loss)(model,batch,context.state)
        torch.testing.assert_close(actual.loss,expected.loss,rtol=1e-5,atol=1e-6)
        assert actual.weight==expected.weight
        actual.loss.backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,gradients[name],rtol=2e-4,atol=1e-6)
    finally:
        context.tracker.close()


def test_named_protocol_rejects_wrong_loss_and_invalid_sample_counts(tmp_path):
    config = configuration(tmp_path,'protocol')
    assert load_config(config=config)['training']['loss']['name']=='normalized_bpr'
    config['training']['loss'] = dict(name='nll')
    with pytest.raises(ValueError,match='training.loss'):
        load_config(config=config)
    config['training']['loss'] = dict(negative_samples=0)
    with pytest.raises(ValueError,match='negative_samples'):
        load_config(config=config)


def test_actual_cpu_amp_training(tmp_path):
    strnn_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0


def test_spawn_worker_can_collate_candidate_geometry(tmp_path):
    strnn_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'spawn')
    config['training'].update(epochs=1,num_workers=1)
    config['evaluation'] = dict(num_workers=1)
    assert run_experiment('train',config=config).metrics['loss']>=0
