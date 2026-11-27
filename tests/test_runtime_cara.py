"""Author-compatible gates, causal candidates and independent BPR pairs."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config
from libcity.data.dataset.trajectory_encoder.stan_encoder import haversine
from libcity.executor.cara_step import pairwise_step
from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.CARA import Recommender as ReferenceRecommender
from libcity.model.trajectory_loc_prediction.runtime_cara import CARACell
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import strnn_fixture,task_fixture
from runtime_helpers import make_context


def configuration(root,experiment):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='CARA',params=dict(latent_dim=4,candidate_chunk_size=3)),
        training=dict(epochs=3,batch_size=8,shuffle=True,accumulation_steps=2,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


def author_reference(model):
    reference = ReferenceRecommender(model.U_Embedding.num_embeddings,model.V_Embedding.num_embeddings,
        model.T_Embedding.num_embeddings,4,device=torch.device('cpu'))
    # The author's Keras notebook has tanh activation and hard_sigmoid gates.
    # LibCity interchanged them. Use its otherwise identical recurrence as an
    # independent reference after explicitly correcting that transcription.
    reference.rnn[0].activation = torch.tanh
    del reference.rnn[0].inner_activation
    reference.rnn[0].inner_activation = lambda value:torch.clamp(.2*value+.5,min=0,max=1)
    mapping = {name:getattr(reference,name).weight for name in ('U_Embedding','V_Embedding','T_Embedding')}
    mapping = {name+'.weight':value for name,value in mapping.items()}
    mapping.update({'cell.'+name:value for name,value in reference.rnn[0].named_parameters()})
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            mapping[name].copy_(parameter)
    return reference,mapping


def test_author_gate_activation_and_full_candidate_reference_scores_and_gradients(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference'))
    try:
        model = context.model
        reference,mapping = author_reference(model)
        batch = next(iter(context.bundle.train))
        batch = {key:value[:2] for key,value in batch.items()}
        batch['query_offset_hours'] = torch.tensor([.25,.5])
        actual = model(context.loss_fn.inputs(batch))
        coordinates = context.bundle.metadata.info['coordinates']+context.bundle.metadata.info['coordinate_origin']
        expected = []
        for index,length in enumerate(batch['lengths']):
            count = int(length)
            prefix = batch['current_loc'][index,:count].tolist()
            times = batch['current_tim'][index,:count].tolist()
            gaps = batch['time_gaps'][index,:count].tolist()
            query_week = float(batch['last_week_hour'][index]+batch['query_offset_hours'][index])%168
            query_time = int(query_week%24)+24*(query_week>=120)
            sequences = [prefix+[candidate+1] for candidate in range(8)]
            distances = []
            for sequence in sequences:
                points = coordinates[np.asarray(sequence)-1]
                distances.append([0.]+[haversine(*a,*b)*6373/6371 for a,b in zip(points[:-1],points[1:])])
            expected.append(reference.rank(np.full(8,int(batch['uid'][index])),np.asarray(sequences),
                np.asarray([times+[query_time]]*8),np.asarray([gaps+[float(batch['query_offset_hours'][index])]]*8),np.asarray(distances)))
        expected = torch.stack(expected)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-7)
        selected = torch.tensor([[0,3],[1,5]])
        actual_loss = context.loss_fn.pairs(actual.gather(1,selected),torch.ones(2,dtype=torch.bool)).loss
        chosen = expected.gather(1,selected)
        expected_loss = (1-torch.log(torch.sigmoid(chosen[:,0]-chosen[:,1]))).mean()
        torch.testing.assert_close(actual_loss,expected_loss)
        actual_loss.backward();expected_loss.backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,mapping[name].grad,rtol=3e-4,atol=2e-7)
        gate = CARACell.gate(torch.tensor([-5.,-2.5,0.,2.5,5.]))
        torch.testing.assert_close(gate,torch.tensor([0.,0.,.5,1.,1.]))
        assert all(not hasattr(module,'device') and not hasattr(module,'_scaler') for module in model.modules())
    finally:
        context.tracker.close()


def test_full_bpr_training_full_ranking_and_exact_rng_resume(tmp_path):
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
    assert expected['train_state']==actual['train_state'] and expected['scheduler']==actual['scheduler']
    torch.testing.assert_close(expected['rng']['torch'],actual['rng']['torch'],rtol=0,atol=0)
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
    records = json.loads(evaluated.artifacts['predictions'].read_text())
    assert records and all(len(row['prediction'])==8 and row['truth'] in range(10,18) for row in records)
    info = json.loads(evaluated.artifacts['runtime_info'].read_text())
    assert info['benchmark_protocol']=='cara_ranking'


def test_padding_batch_chunk_float64_and_input_label_isolation(tmp_path):
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
        poisoned['current_tim'][~valid] = -999
        poisoned['time_gaps'][~valid] = float('nan')
        poisoned.update(y=torch.full_like(batch['y'],9999),target_tim=torch.tensor(float('nan')))
        rng = torch.get_rng_state().clone()
        torch.testing.assert_close(model(poisoned),prediction,rtol=0,atol=0)
        model.chunk_size = 1
        torch.testing.assert_close(model(inputs),prediction,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        assert prediction.dtype==torch.float64 and prediction.shape==(len(batch['y']),8)
        for key,value in before.items():
            torch.testing.assert_close(inputs[key],value,rtol=0,atol=0)
        context.loss_fn(prediction,batch['y']).loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        with pytest.raises(ValueError,match='query_offset_hours'):
            model({**inputs,'query_offset_hours':torch.full((len(batch['y']),),-1.)})
    finally:
        context.tracker.close()


def test_real_timestamp_gaps_across_weekend_midnight_and_known_query_clock(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    indices = frame.index[frame.traj_id==17]
    start = pd.Timestamp('2024-01-20T23:59:30Z')
    frame.loc[indices,'time'] = [(start+pd.Timedelta(minutes=index)).strftime('%Y-%m-%dT%H:%M:%SZ') for index in range(len(indices))]
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    context = make_context(configuration(tmp_path,'midnight'))
    try:
        rows = [row for loader in (context.bundle.train,context.bundle.val,context.bundle.test) for row in loader.dataset.rows]
        first = next(row for row in rows if int(row['sample_id'])==int(frame.loc[indices[1],'dyna_id']))
        second = next(row for row in rows if int(row['sample_id'])==int(frame.loc[indices[2],'dyna_id']))
        torch.testing.assert_close(second['current_tim'],torch.tensor([47,24]))
        torch.testing.assert_close(second['time_gaps'],torch.tensor([0.,1/60]))
        batch = context.bundle.test.collate_fn([first])
        base = context.model.encode(context.loss_fn.inputs(batch))
        later = context.model.encode({**context.loss_fn.inputs(batch),'query_offset_hours':torch.tensor([1/60])})
        torch.testing.assert_close(base[2],context.model.T_Embedding(torch.tensor([47])))
        torch.testing.assert_close(later[2],context.model.T_Embedding(torch.tensor([24])))
    finally:
        context.tracker.close()


def test_changed_real_next_time_and_location_leave_inputs_and_prediction_unchanged(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'features'))
    try:
        sample = context.bundle.test.dataset.rows[-1]
        identity = int(sample['sample_id'])
        batch = context.bundle.test.collate_fn([sample])
        prediction = context.model.eval()(context.loss_fn.inputs(batch))
        state = {key:value.clone() for key,value in context.model.state_dict().items()}
    finally:
        context.tracker.close()
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    index = frame.index[frame.dyna_id==identity][0]
    frame.loc[index,'time'] = (pd.Timestamp(frame.loc[index,'time'])+pd.Timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.loc[index,'location'] = 10+(int(frame.loc[index,'location'])-10+3)%8
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    changed = make_context(configuration(tmp_path,'changed'))
    try:
        changed.model.eval().load_state_dict(state)
        row = next(row for row in changed.bundle.test.dataset.rows if int(row['sample_id'])==identity)
        after = changed.bundle.test.collate_fn([row])
        for key,value in changed.loss_fn.inputs(batch).items():
            torch.testing.assert_close(after[key],value,rtol=0,atol=0)
        torch.testing.assert_close(changed.model(changed.loss_fn.inputs(after)),prediction,rtol=0,atol=0)
    finally:
        changed.tracker.close()


def test_bpr_is_per_sample_and_negatives_exclude_history_positive_with_finite_fallback(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'pairs'))
    try:
        loss = context.loss_fn
        scores = torch.tensor([[4.,3.],[-2.,-3.]],requires_grad=True)
        actual = loss.pairs(scores,torch.ones(2,dtype=torch.bool)).loss
        expected = torch.stack([1-F.logsigmoid(row[0]-row[1]) for row in scores]).mean()
        torch.testing.assert_close(actual,expected)
        assert not torch.isclose(actual,1-F.logsigmoid((scores[:,0]-scores[:,1]).sum()))
        actual.backward()
        torch.testing.assert_close(scores.grad[0],scores.grad[1])
        batch = dict(current_loc=torch.tensor([[1,2,0],[1,2,3]]),lengths=torch.tensor([2,3]),y=torch.tensor([2,3]))
        for _ in range(10):
            negative,valid = loss.sample(batch)
            assert valid.all() and int(negative[0]) in range(3,8) and int(negative[1]) in range(4,8)
        exhausted = dict(current_loc=torch.arange(1,9)[None],lengths=torch.tensor([8]),y=torch.tensor([0]))
        negative,valid = loss.sample(exhausted)
        assert bool(valid[0]) and int(negative[0]) in range(1,8)
    finally:
        context.tracker.close()


def test_selected_training_matches_full_scores_gradients_and_encodes_prefix_once(tmp_path,monkeypatch):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'selected'))
    try:
        batch = next(iter(context.bundle.train))
        model,loss = context.model,context.loss_fn
        rng = torch.get_rng_state().clone()
        negative,valid = loss.sample(batch)
        candidates = torch.stack((batch['y'],negative),dim=-1)
        expected = loss.pairs(model(loss.inputs(batch)).gather(1,candidates),valid)
        expected.loss.backward()
        gradients = {name:parameter.grad.clone() for name,parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        calls = []
        encode = model.encode
        def counted(*args,**kwargs):
            calls.append(1)
            return encode(*args,**kwargs)
        def forbidden(*args,**kwargs):
            raise AssertionError('sampled training must not call full-vocabulary forward')
        monkeypatch.setattr(model,'encode',counted)
        monkeypatch.setattr(model,'forward',forbidden)
        actual = pairwise_step(loss)(model,batch,context.state)
        assert len(calls)==1
        torch.testing.assert_close(actual.loss,expected.loss,rtol=1e-5,atol=1e-6)
        actual.loss.backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,gradients[name],rtol=3e-4,atol=2e-7)
    finally:
        context.tracker.close()


def test_long_prefix_finite_and_explicit_recent_window_matches_cropped_context(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'long'))
    try:
        model = context.model
        length = 128
        batch = dict(current_loc=(torch.arange(length)%8+1)[None],current_tim=(torch.arange(length)%24)[None],
            time_gaps=torch.ones(1,length),lengths=torch.tensor([length]),uid=torch.tensor([0]),last_week_hour=torch.tensor([127.]))
        prediction = model(batch)
        assert torch.isfinite(prediction).all()
        context.loss_fn(prediction,torch.tensor([0])).loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        model.history_window = 5
        windowed = model(batch)
        cropped = {key:value[:,-5:] if value.ndim==2 else value for key,value in batch.items()}
        cropped['lengths'] = torch.tensor([5])
        model.history_window = 0
        torch.testing.assert_close(model(cropped),windowed,rtol=0,atol=0)
    finally:
        context.tracker.close()


def test_single_location_without_any_negative_skips_updates_and_evaluates(tmp_path):
    folder = task_fixture(tmp_path/'raw')
    geo = pd.read_csv(folder/'TaskTiny.geo').iloc[:1];geo.to_csv(folder/'TaskTiny.geo',index=False)
    frame = pd.read_csv(folder/'TaskTiny.dyna');frame['location'] = 10
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    config = configuration(tmp_path,'single')
    config['training']['epochs'] = 1
    result = run_experiment('train',config=config)
    assert result.metrics['Recall@1']==1. and result.metrics['loss']==0.
    state = torch.load(result.artifacts['last'],weights_only=True)
    assert state['train_state']['optimizer_steps']==0


def test_geometry_uses_longitude_latitude_radians_fractional_kilometres(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'geometry'))
    try:
        model = context.model.double()
        point = model.coordinates
        actual = model.distances(point[0],point)
        absolute = (point+model.coordinate_origin).detach().numpy()
        expected = torch.tensor([haversine(*absolute[0],*other)*6373/6371 for other in absolute],dtype=torch.float64)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-9)
        assert bool((actual[1:]>0).all()) and bool((actual[1:]<1).all())
    finally:
        context.tracker.close()


def test_protocol_rejects_cross_entropy_in_place_of_bpr(tmp_path):
    config = configuration(tmp_path,'protocol')
    assert load_config(config=config)['training']['loss']['name']=='bpr'
    config['training']['loss'] = dict(name='nll')
    with pytest.raises(ValueError,match='training.loss'):
        load_config(config=config)


def test_actual_cpu_amp_training(tmp_path):
    strnn_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0
