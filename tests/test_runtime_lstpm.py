"""LSTPM reference gradients, causal features and complete ranking lifecycle."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.data.dataset.trajectory_encoder.lstpm_encoder import geodistance
from libcity.data.lstpm_trajectory import spatial_parents,time_similarity
from libcity.data.task_bundles import collate_sequences,prepare_history_trajectory
from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.LSTPM import LSTPM as ReferenceLSTPM
from libcity.model.trajectory_loc_prediction.runtime_lstpm import LSTPM
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import strnn_fixture,task_fixture
from runtime_helpers import make_context


def configuration(root,name):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='LSTPM',params=dict(emb_size=3,hidden_size=4)),
        training=dict(epochs=2,batch_size=8,shuffle=True,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


class ReferenceInput(dict):
    def get_origin_len(self,key):
        return self['lengths'].tolist()


def test_original_two_level_attention_dilated_recurrence_and_all_gradients(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference'))
    try:
        rng = np.random.default_rng(17)
        similarity = rng.uniform(0,1,(49,49)).astype(np.float32)
        similarity = (similarity+similarity.T)/2
        similarity[48] = similarity[:,48] = 0.
        metadata = replace(context.bundle.metadata,info={**context.bundle.metadata.info,'tim_sim_matrix':similarity})
        model = LSTPM(metadata=metadata,emb_size=3,hidden_size=4)
        reference = ReferenceLSTPM(dict(emb_size=3,hidden_size=4,dropout=.8,device=torch.device('cpu')),
            {**metadata.info,'tim_sim_matrix':similarity.tolist()})
        missing = model.load_state_dict(reference.state_dict(),strict=False)
        assert missing.missing_keys==['tim_sim_matrix'] and not missing.unexpected_keys
        batch = dict(current_loc=torch.tensor([[1,2,0],[3,4,5]]),current_tim=torch.tensor([[1,2,0],[3,4,5]]),
            lengths=torch.tensor([2,3]),dilated_rnn_input_index=torch.tensor([[0,0,0],[0,0,1]]),
            history_loc=torch.tensor([[2,3,4,0],[4,5,6,7]]),history_tim=torch.tensor([[6,7,8,0],[9,10,11,12]]),
            history_lengths=torch.tensor([3,4]),history_session_lengths=torch.tensor([[2,1],[3,1]]),
            history_avg_distance=torch.tensor([[1.2,2.7],[2.5,1.]]))
        old = ReferenceInput(batch)
        old['history_loc'] = [[torch.tensor([2,3]),torch.tensor([4])],[torch.tensor([4,5,6]),torch.tensor([7])]]
        old['history_tim'] = [[torch.tensor([6,7]),torch.tensor([8])],[torch.tensor([9,10,11]),torch.tensor([12])]]
        old['dilated_rnn_input_index'] = [torch.tensor([0,0]),torch.tensor([0,0,1])]
        expected = reference(old)[:,1:]
        expected = expected-expected.logsumexp(-1,keepdim=True)
        actual = model(batch)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        target = torch.tensor([2,5])
        F.nll_loss(expected,target).backward();F.nll_loss(actual,target).backward()
        parameters = dict(reference.named_parameters())
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,parameters[name].grad,rtol=1e-4,atol=1e-6)
        assert 'tim_sim_matrix' in dict(model.named_buffers())
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in model.modules())
    finally:
        context.tracker.close()


def test_causal_completed_sessions_centroids_and_preserved_cold_start_samples(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'features'))
    try:
        cfg = context.config
        base = prepare_history_trajectory(cfg['data'],cfg['training'],cfg['evaluation'],17)
        frame = pd.read_csv(folder/'TaskTiny.dyna');frame['time'] = pd.to_datetime(frame['time'],utc=True)
        coordinates = np.asarray([json.loads(value) for value in pd.read_csv(folder/'TaskTiny.geo')['coordinates']])
        groups = [group.sort_values('time') for _,group in frame.groupby(['entity_id','traj_id'])]
        for split in ('train','val','test'):
            rows = getattr(context.bundle,split).dataset.rows
            assert [int(row['sample_id']) for row in rows]==[int(row['sample_id']) for row in getattr(base,split).dataset.rows]
            for row in rows:
                target = frame.loc[frame.dyna_id==int(row['sample_id'])].iloc[0]
                current = frame[(frame.entity_id==target.entity_id)&(frame.traj_id==target.traj_id)]
                previous = sorted([group for group in groups if group.entity_id.iloc[0]==target.entity_id
                    and group.time.max()<current.time.min()],key=lambda group:group.time.max())
                assert row['history_session_lengths'].tolist()==[len(group) for group in previous]
                last = coordinates[int(row['current_loc'][-1])-1]
                expected = [max(1.,geodistance(*(coordinates[group.location.to_numpy()-10].mean(0)[::-1]),*last[::-1]))
                    for group in previous]
                torch.testing.assert_close(row['history_avg_distance'],torch.tensor(expected,dtype=torch.float32))
        assert sum(context.bundle.split_summary['cold_start_samples'].values())==5
        sample = next(row for row in context.bundle.train.dataset.rows if int(row['history_lengths'])>0)
        saved = {key:value.clone() for key,value in sample.items()}
        batch = collate_sequences([sample]);batch['history_session_lengths'].zero_();batch['dilated_rnn_input_index'].fill_(999)
        for key,value in saved.items(): torch.testing.assert_close(value,sample[key],rtol=0,atol=0)
    finally:
        context.tracker.close()
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    frame.loc[frame.loc[frame.traj_id==0].index[-1],'time'] = '2024-01-03T00:30:00Z'
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    overlap = make_context(configuration(tmp_path,'overlap'))
    try:
        ids = set(frame.loc[frame.traj_id==2,'dyna_id'])
        rows = [row for loader in (overlap.bundle.train,overlap.bundle.val,overlap.bundle.test)
            for row in loader.dataset.rows if int(row['sample_id']) in ids]
        assert rows and all(not len(row['history_session_lengths']) and int(row['history_lengths'])==0 for row in rows)
    finally:
        overlap.tracker.close()


def test_time_similarity_fits_atomic_training_points_only_including_empty_slots():
    group = pd.DataFrame({'time':pd.to_datetime(['2024-01-01T01:00Z','2024-01-01T02:00Z','2024-01-02T02:00Z',
        '2024-01-02T03:00Z','2024-01-06T01:00Z','2024-01-08T01:00Z'],utc=True)})
    locations = [1,1,2,3,2,99]
    cut = pd.Timestamp('2024-01-08',tz='UTC').value
    actual,count = time_similarity([(group,locations)],cut)
    assert count==5 and actual.shape==(49,49)
    assert actual[1,2]==.5 and actual[2,25]==.5 and actual[1,25]==0.
    np.testing.assert_array_equal(actual,actual.T)
    np.testing.assert_array_equal(np.diag(actual)[:48],np.ones(48))
    assert not actual[48].any() and not actual[:,48].any() and actual[0,4]==0.
    changed,_ = time_similarity([(group,[*locations[:-1],1])],cut)
    np.testing.assert_array_equal(actual,changed)


def test_spatial_parents_match_original_rounding_ties_and_every_prefix():
    coordinates = np.array([[0.,0.],[1.,0.],[0.,0.],[0.,0.],[179.,89.],[-179.,89.],[.00000001,0.]])
    before = coordinates.copy()
    actual = spatial_parents(coordinates)
    expected = [0]
    for position in range(1,len(coordinates)):
        distances = [geodistance(*coordinates[position,::-1],*coordinates[earlier,::-1]) for earlier in range(position)]
        minimum = min(distances)
        expected.append(max(index for index,value in enumerate(distances) if value==minimum))
    assert actual.tolist()==expected and actual[:4].tolist()==[0,0,0,2]
    for stop in range(1,len(coordinates)):
        torch.testing.assert_close(actual[:stop],spatial_parents(coordinates[:stop]))
    np.testing.assert_array_equal(coordinates,before)


def test_full_training_ranking_export_and_exact_resume(tmp_path):
    strnn_fixture(tmp_path/'raw')
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,until_epoch=1);partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert actual['train_state']==expected['train_state'] and actual['scheduler']==expected['scheduler']
    for name,value in expected['model'].items(): torch.testing.assert_close(value,actual['model'][name],rtol=0,atol=0)
    records = json.loads(evaluated.artifacts['predictions'].read_text())
    assert records and all(len(row['prediction'])==8 and row['truth'] in range(10,18) for row in records)


@pytest.mark.parametrize('all_empty',[False,True])
def test_padding_mixed_batches_float64_and_empty_history_are_finite(tmp_path,all_empty):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'padding'))
    try:
        rows = context.bundle.train.dataset.rows
        cold = next(row for row in rows if int(row['history_lengths'])==0)
        other = next(row for row in rows if int(row['history_lengths'])>0)
        chosen = [cold,cold] if all_empty else [cold,other]
        inputs = context.loss_fn.inputs(collate_sequences(chosen))
        saved = {key:value.clone() for key,value in inputs.items()}
        model = context.model.eval().double()
        state = {key:value.clone() for key,value in model.state_dict().items()}
        rng = torch.get_rng_state()
        actual = model(inputs)
        assert actual.shape==(2,8) and actual.dtype==torch.float64 and torch.isfinite(actual).all()
        for index,row in enumerate(chosen):
            individual = context.loss_fn.inputs(collate_sequences([row]))
            torch.testing.assert_close(model(individual),actual[index:index+1],rtol=1e-12,atol=1e-12)
        poisoned = {key:F.pad(value,(0,3),value=99999) if value.ndim==2 else value.clone() for key,value in inputs.items()}
        # Session lengths define boundaries: unlike token padding, they must remain zero.
        poisoned['history_session_lengths'][:,-3:] = 0
        poisoned['y'] = torch.tensor([9999,9999])
        torch.testing.assert_close(model(poisoned),actual,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(actual.exp().sum(-1),torch.ones(2,dtype=torch.float64))
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key,value in saved.items(): torch.testing.assert_close(inputs[key],value,rtol=0,atol=0)
        for key,value in state.items(): torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        actual.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        context.tracker.close()


def test_real_next_event_never_changes_prediction_inputs_or_time_fit(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    first = make_context(configuration(tmp_path,'before'))
    try:
        row = first.bundle.test.dataset.rows[-1]
        saved = {key:value.clone() for key,value in row.items()}
        before = first.model(first.loss_fn.inputs(collate_sequences([row])))
        frame = pd.read_csv(folder/'TaskTiny.dyna')
        selected = frame.index[frame.dyna_id==int(row['sample_id'])][0]
        frame.loc[selected,'location'] = 10+(int(frame.loc[selected,'location'])-9)%8
        frame.loc[selected,'time'] = (pd.Timestamp(frame.loc[selected,'time'])+pd.Timedelta(seconds=30)).isoformat()
        frame.to_csv(folder/'TaskTiny.dyna',index=False)
        second = make_context(configuration(tmp_path,'after'))
        try:
            new = next(item for item in second.bundle.test.dataset.rows if int(item['sample_id'])==int(row['sample_id']))
            assert int(saved['y'])!=int(new['y'])
            for key,value in first.loss_fn.inputs(saved).items(): torch.testing.assert_close(value,new[key],rtol=0,atol=0)
            np.testing.assert_array_equal(first.bundle.metadata.info['tim_sim_matrix'],second.bundle.metadata.info['tim_sim_matrix'])
            torch.testing.assert_close(before,second.model(second.loss_fn.inputs(collate_sequences([new]))),rtol=0,atol=0)
        finally:
            second.tracker.close()
    finally:
        first.tracker.close()


@pytest.mark.parametrize('problem',['parent','lengths','sessions','distance','time'])
def test_invalid_sequence_contracts_fail_explicitly(tmp_path,problem):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'invalid'))
    try:
        row = next(row for row in context.bundle.train.dataset.rows if int(row['history_lengths'])>0 and int(row['lengths'])>1)
        inputs = context.loss_fn.inputs(collate_sequences([row]))
        if problem=='parent': inputs['dilated_rnn_input_index'][0,1] = 1
        elif problem=='lengths': inputs['lengths'].zero_()
        elif problem=='sessions': inputs['history_session_lengths'][0,0] += 1
        elif problem=='distance': inputs['history_avg_distance'][0,0] = float('nan')
        else: inputs['current_tim'][0,0] = 48
        with pytest.raises(ValueError): context.model(inputs)
    finally:
        context.tracker.close()


def test_actual_cpu_amp_training_and_identical_coordinates(tmp_path):
    task_fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'amp')
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all() and result.metrics['loss']>=0
    checkpoint = torch.load(result.artifacts['last'],weights_only=True)
    assert checkpoint['train_state']['optimizer_steps']>0
