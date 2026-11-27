"""Every candidate uses its own geometry and only known prefix timestamps."""
import json

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.STRNN import STRNN as ReferenceSTRNN
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import task_fixture,strnn_fixture
from runtime_helpers import make_context


def configuration(root,experiment):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='STRNN',params=dict(hidden_size=4,candidate_chunk_size=3)),
        training=dict(epochs=2,batch_size=8,shuffle=True,scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


def candidate_reference(reference,model,batch):
    elapsed,locations,lengths = batch['elapsed_hours'],batch['current_loc'],batch['lengths']
    query = elapsed.gather(1,(lengths-1)[:,None])
    temporal = (query-elapsed).clamp(0,model.up_time)
    observed = model.coordinates[(locations-1).clamp_min(0)]
    user = reference.permanet_weight(batch['uid'])
    scores = []
    for candidate in range(len(model.coordinates)):
        distance = (model.coordinates[candidate]-observed).square().sum(-1).sqrt().clamp(0,model.up_loc)
        state = reference(model.up_time-temporal,temporal,model.up_loc-distance,distance,locations,lengths.tolist())
        scores.append(((state+user)*reference.location_weight.weight[candidate+1]).sum(-1))
    return torch.stack(scores,dim=1)


def test_original_transition_matrices_per_candidate_scores_and_gradients(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference'))
    try:
        model = context.model
        metadata = dict(context.bundle.metadata.info,tim_size=model.up_time+1,distance_upper=model.up_loc)
        reference = ReferenceSTRNN(dict(hidden_size=4,device=torch.device('cpu')),metadata)
        missing = model.load_state_dict(reference.state_dict(),strict=False)
        assert missing.missing_keys==['coordinates'] and not missing.unexpected_keys
        batch = next(iter(context.bundle.train))
        expected_scores = candidate_reference(reference,model,batch)
        actual_scores = model.scores(context.loss_fn.inputs(batch))
        torch.testing.assert_close(actual_scores,expected_scores,rtol=1e-5,atol=1e-6)
        expected = F.log_softmax(expected_scores,dim=-1)
        actual = model(context.loss_fn.inputs(batch))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        F.nll_loss(expected,batch['y']).backward()
        context.loss_fn(actual,batch['y']).loss.backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-4,atol=1e-6)
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in model.modules())
    finally:
        context.tracker.close()


def test_complete_training_ranking_export_and_exact_resume(tmp_path):
    strnn_fixture(tmp_path/'raw')
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert expected['train_state']==actual['train_state']
    assert expected['scheduler']==actual['scheduler']
    for name,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
    records = json.loads(evaluated.artifacts['predictions'].read_text())
    assert records and all(len(row['prediction'])==8 and row['truth'] in range(10,18) for row in records)


def test_padding_chunking_batch_independence_and_target_isolation_float64(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'inputs'))
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        inputs = context.loss_fn.inputs(batch)
        inputs['elapsed_hours'] = inputs['elapsed_hours'].double()
        before = {key:value.clone() for key,value in inputs.items()}
        rng = torch.get_rng_state()
        prediction = model(inputs)
        model.chunk_size = 1
        torch.testing.assert_close(prediction,model(inputs),rtol=1e-10,atol=1e-10)
        one = {key:value[:1] for key,value in inputs.items()}
        torch.testing.assert_close(prediction[:1],model(one),rtol=1e-10,atol=1e-10)
        extended = {key:F.pad(value,(0,3),value=9999) if value.ndim==2 else value for key,value in one.items()}
        extended.update(y=torch.full((1,),9999),target_tim=torch.full((1,),float('nan')),current_dis=torch.full((1,1),float('nan')))
        torch.testing.assert_close(prediction[:1],model(extended),rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(prediction.exp().sum(-1),torch.ones(len(prediction),dtype=torch.float64))
        assert prediction.dtype==torch.float64 and prediction.shape==(len(batch['y']),8)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key,value in before.items():
            torch.testing.assert_close(value,inputs[key],rtol=0,atol=0)
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        with pytest.raises(ValueError,match='query_offset_hours'):
            model({**one,'query_offset_hours':torch.tensor([-1.])})
    finally:
        context.tracker.close()


def test_geometry_changes_candidate_state_even_with_identical_candidate_embeddings(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'geometry'))
    try:
        model = context.model.eval()
        with torch.no_grad():
            model.location_weight.weight.fill_(1.)
            model.permanet_weight.weight.zero_()
            model.h0.zero_();model.weight_ih.zero_()
            model.weight_th_upper.copy_(torch.eye(4));model.weight_th_lower.copy_(torch.eye(4))
            model.weight_sh_upper.copy_(torch.eye(4));model.weight_sh_lower.copy_(2*torch.eye(4))
        inputs = dict(current_loc=torch.tensor([[1]]),elapsed_hours=torch.tensor([[0.]]),lengths=torch.tensor([1]),uid=torch.tensor([0]))
        scores = model.scores(inputs)[0]
        ratios = model.coordinates.square().sum(-1).sqrt().div(model.up_loc).clamp(0,1)
        torch.testing.assert_close(scores,4*torch.sigmoid(1+ratios))
        assert scores.max()>scores.min() and scores.argmin()==0
    finally:
        context.tracker.close()


def test_prefix_features_and_training_bounds_do_not_use_next_event_time(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'features'))
    try:
        observed_upper = max(float(row['elapsed_hours'][-1]-row['elapsed_hours'][0]) for row in context.bundle.train.dataset.rows)
        assert context.model.up_time==observed_upper
        assert context.bundle.split_summary['time_upper_fitted_on']=='training_prefixes'
        frame = pd.read_csv(folder/'TaskTiny.dyna')
        frame['time'] = pd.to_datetime(frame['time'],utc=True)
        groups = {int(identity):part.sort_values('time') for identity,part in frame.groupby('traj_id')}
        for loader in (context.bundle.train,context.bundle.val,context.bundle.test):
            for row in loader.dataset.rows:
                target = frame[frame.dyna_id==int(row['sample_id'])].iloc[0]
                trip = groups[int(target.traj_id)]
                prefix = trip.iloc[:int(row['lengths'])]
                expected = torch.tensor((prefix.time-prefix.time.iloc[0]).dt.total_seconds().to_numpy()/3600,dtype=torch.float32)
                torch.testing.assert_close(row['elapsed_hours'],expected,rtol=0,atol=0)
        sample = context.bundle.test.dataset.rows[-1]
        target_id = int(sample['sample_id'])
        model = context.model.eval()
        prediction = model({key:value[None] for key,value in sample.items()})
        weights = {key:value.clone() for key,value in model.state_dict().items()}
    finally:
        context.tracker.close()
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    point = frame.index[frame.dyna_id==target_id][0]
    frame.loc[point,'time'] = (pd.Timestamp(frame.loc[point,'time'])+pd.Timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    changed = make_context(configuration(tmp_path,'changed-target-time'))
    try:
        changed.model.eval().load_state_dict(weights)
        sample = next(row for row in changed.bundle.test.dataset.rows if int(row['sample_id'])==target_id)
        torch.testing.assert_close(prediction,changed.model({key:value[None] for key,value in sample.items()}),rtol=0,atol=0)
    finally:
        changed.tracker.close()


def test_coincident_coordinates_and_zero_time_spans_have_finite_bounds(tmp_path):
    folder = task_fixture(tmp_path/'raw')
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    frame['time'] = frame.groupby('traj_id')['time'].transform('first')
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    context = make_context(configuration(tmp_path,'degenerate'))
    try:
        assert context.model.up_time==context.model.up_loc==1.
        batch = next(iter(context.bundle.train))
        prediction = context.model(context.loss_fn.inputs(batch))
        assert torch.isfinite(prediction).all()
        context.loss_fn(prediction,batch['y']).loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in context.model.parameters())
    finally:
        context.tracker.close()


def test_invalid_coordinates_fail_at_data_entry(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    geo = pd.read_csv(folder/'TaskTiny.geo');geo.loc[0,'coordinates'] = '[[1,2],[3,4]]'
    geo.to_csv(folder/'TaskTiny.geo',index=False)
    with pytest.raises(ValueError,match='point coordinates'):
        make_context(configuration(tmp_path,'invalid'))


def test_actual_cpu_amp_training(tmp_path):
    strnn_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0
