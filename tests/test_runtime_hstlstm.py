"""Memory recurrence, causal interval features and the shared HSTLSTM lifecycle."""
from dataclasses import replace
import json
import math

from geopy.distance import geodesic
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.executor.train import train_loop
from libcity.model.trajectory_loc_prediction.HSTLSTM import HSTLSTM as ReferenceHSTLSTM
from libcity.model.trajectory_loc_prediction.runtime_hstlstm import HSTLSTM,STLSTMCell
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import task_fixture,strnn_fixture
from runtime_helpers import make_context


def configuration(root,experiment):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='HSTLSTM',params=dict(tim_slot_len=7,dis_slot_len=5,embed_size=3,hidden_size=4)),
        training=dict(epochs=2,batch_size=8,shuffle=True,scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


def corrected_cell(cell,location,temporal,spatial,state):
    hidden,memory = state
    gates = location@cell.w_ih.T+hidden@cell.w_hh.T+cell.b_ih+cell.b_hh
    incoming,forget,candidate,outgoing = gates.chunk(4,-1)
    ti,tf,to = (temporal@cell.w_s.T+spatial@cell.w_q.T).chunk(3,-1)
    next_memory = torch.sigmoid(forget+tf)*memory+torch.sigmoid(incoming+ti)*torch.tanh(candidate)
    return torch.sigmoid(outgoing+to)*torch.tanh(next_memory),next_memory


def test_memory_recurrence_matches_independent_gates_and_depends_on_previous_cell():
    torch.manual_seed(17)
    cell = STLSTMCell(3,4).double()
    inputs = [torch.randn(2,width,dtype=torch.float64,requires_grad=True) for width in (3,3,3,4,4)]
    expected = corrected_cell(cell,*inputs[:3],tuple(inputs[3:]))
    actual = cell(*inputs[:3],tuple(inputs[3:]))
    for value,reference in zip(actual,expected):
        torch.testing.assert_close(value,reference,rtol=1e-12,atol=1e-12)
    variables = inputs+list(cell.parameters())
    expected_grad = torch.autograd.grad(sum(item.square().sum() for item in expected),variables,retain_graph=True)
    actual_grad = torch.autograd.grad(sum(item.square().sum() for item in actual),variables)
    for value,reference in zip(actual_grad,expected_grad):
        torch.testing.assert_close(value,reference,rtol=1e-12,atol=1e-12)
    with torch.no_grad():
        for parameter in cell.parameters():
            parameter.zero_()
    zero = torch.zeros(1,3,dtype=torch.float64)
    hidden,memory = cell(zero,zero,zero,(torch.zeros(1,4,dtype=torch.float64),torch.full((1,4),2.,dtype=torch.float64)))
    torch.testing.assert_close(memory,torch.ones_like(memory))
    torch.testing.assert_close(hidden,.5*torch.ones_like(hidden).tanh())


def test_original_classifier_and_gradients_with_explicitly_corrected_memory(tmp_path):
    strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference'))
    try:
        metadata = replace(context.bundle.metadata,info={**context.bundle.metadata.info,'tim_slot_max':600,'dis_slot_max':4})
        model = HSTLSTM(metadata=metadata,**context.config['model']['params'])
        reference = ReferenceHSTLSTM(dict(tim_slot_len=7,dis_slot_len=5,embed_size=3,hidden_size=4,device=torch.device('cpu')),metadata.info)
        missing = model.load_state_dict(reference.state_dict(),strict=False)
        assert set(missing.missing_keys)=={'tim_slots','dis_slots'} and not missing.unexpected_keys
        def forward(location,temporal,spatial,hc=None):
            if hc is None:
                zero = location.new_zeros(len(location),4)
                hc = (zero,zero)
            return corrected_cell(reference.st_lstm.cell,location,temporal,spatial,hc)
        reference.st_lstm.cell.forward = forward
        batch = next(iter(context.bundle.train))
        expected = reference(batch['current_loc'],batch['tim_interval'],batch['dis'],batch['lengths'].tolist())[:,1:]
        expected = expected-torch.logsumexp(expected,dim=1,keepdim=True)
        actual = model(context.loss_fn.inputs(batch))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        F.nll_loss(expected,batch['y']).backward()
        F.nll_loss(actual,batch['y']).backward()
        parameters = dict(reference.named_parameters())
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,parameters[name].grad,rtol=2e-4,atol=1e-6)
    finally:
        context.tracker.close()


def test_full_training_ranking_export_and_exact_resume(tmp_path):
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


def test_fractional_slots_clamp_exact_boundaries_and_preserve_embedding_gradients():
    embedding = torch.nn.Embedding(3,2).double()
    with torch.no_grad():
        embedding.weight.copy_(torch.tensor([[1.,2.],[3.,7.],[10.,12.]]))
    slots = torch.tensor([0.,.4,1.],dtype=torch.float64)
    values = torch.tensor([[-1.,0.,.1,.4,.7,1.,10.]],dtype=torch.float64)
    actual = HSTLSTM.interpolate(values,slots,embedding)
    expected = torch.stack((embedding.weight[0],embedding.weight[0],.75*embedding.weight[0]+.25*embedding.weight[1],
        embedding.weight[1],.5*(embedding.weight[1]+embedding.weight[2]),embedding.weight[2],embedding.weight[2]))[None]
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    actual.sum().backward()
    assert torch.isfinite(embedding.weight.grad).all() and (embedding.weight.grad>0).all()


def test_padding_batch_independence_float64_and_no_label_or_input_mutation(tmp_path):
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
        poisoned.update(y=torch.full_like(batch['y'],9999),target_tim=torch.tensor(float('nan')))
        torch.testing.assert_close(model(poisoned),prediction,rtol=0,atol=0)
        assert prediction.dtype==torch.float64 and prediction.shape==(len(batch['y']),8)
        torch.testing.assert_close(prediction.exp().sum(-1),torch.ones(len(prediction),dtype=torch.float64))
        assert all(not hasattr(module,'device') and not hasattr(module,'_scaler') for module in model.modules())
        for key,value in before.items():
            torch.testing.assert_close(inputs[key],value,rtol=0,atol=0)
        prediction.square().mean().backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        with pytest.raises(ValueError,match='lengths'):
            model({**inputs,'lengths':torch.zeros_like(inputs['lengths'])})
        with pytest.raises(ValueError,match='nonnegative'):
            model({**inputs,'tim_interval':torch.full_like(inputs['tim_interval'],-1.)})
    finally:
        context.tracker.close()


def test_atomic_intervals_are_geodesic_prefix_only_with_train_fitted_bounds(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'features'))
    try:
        frame = pd.read_csv(folder/'TaskTiny.dyna')
        frame['time'] = pd.to_datetime(frame['time'],utc=True)
        geo = pd.read_csv(folder/'TaskTiny.geo')
        points = {row.geo_id:json.loads(row.coordinates) for row in geo.itertuples()}
        groups = {identity:part.sort_values('time') for identity,part in frame.groupby('traj_id')}
        bundle = context.bundle
        assert bundle.metadata.input_layout['slot_bounds_fitted_on']=='training_prefixes'
        assert bundle.metadata.info['tim_slot_max']==math.ceil(max(100.,max(float(row['tim_interval'].max()) for row in bundle.train.dataset.rows)))
        for loader in (bundle.train,bundle.val,bundle.test):
            for row in loader.dataset.rows:
                target = frame[frame.dyna_id==int(row['sample_id'])].iloc[0]
                prefix = groups[target.traj_id].iloc[:int(row['lengths'])]
                expected_time = torch.tensor(prefix.time.diff().dt.total_seconds().fillna(0).to_numpy(),dtype=torch.float32)
                locations = [points[value] for value in prefix.location]
                expected_dis = torch.tensor([.5]+[geodesic(a[::-1],b[::-1]).km for a,b in zip(locations[:-1],locations[1:])])
                torch.testing.assert_close(row['tim_interval'],expected_time,rtol=0,atol=0)
                torch.testing.assert_close(row['dis'],expected_dis,rtol=0,atol=0)
                assert 'current_tim' not in row
        sample = bundle.test.dataset.rows[-1]
        sample_id = int(sample['sample_id'])
        prediction = context.model.eval()({key:value[None] for key,value in sample.items()})
        state = {key:value.clone() for key,value in context.model.state_dict().items()}
        bounds = bundle.metadata.info['tim_slot_max'],bundle.metadata.info['dis_slot_max']
        collated = bundle.test.collate_fn([sample])
        collated['tim_interval'].add_(1000)
        assert float(sample['tim_interval'][0])==0.
    finally:
        context.tracker.close()
    # Change only the unseen next-event time; the input, bounds and score stay fixed.
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    index = frame.index[frame.dyna_id==sample_id][0]
    frame.loc[index,'time'] = (pd.Timestamp(frame.loc[index,'time'])+pd.Timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    changed = make_context(configuration(tmp_path,'changed'))
    try:
        assert bounds==(changed.bundle.metadata.info['tim_slot_max'],changed.bundle.metadata.info['dis_slot_max'])
        changed.model.eval().load_state_dict(state)
        sample = next(row for row in changed.bundle.test.dataset.rows if int(row['sample_id'])==sample_id)
        torch.testing.assert_close(changed.model({key:value[None] for key,value in sample.items()}),prediction,rtol=0,atol=0)
    finally:
        changed.tracker.close()


def test_zero_intervals_and_coincident_points_remain_finite(tmp_path):
    folder = task_fixture(tmp_path/'raw')
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    frame['time'] = frame.groupby('traj_id')['time'].transform('first')
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    context = make_context(configuration(tmp_path,'zero'))
    try:
        assert context.bundle.metadata.info['tim_slot_max']==100
        assert context.bundle.metadata.info['dis_slot_max']==1
        batch = next(iter(context.bundle.train))
        prediction = context.model(context.loss_fn.inputs(batch))
        context.loss_fn(prediction,batch['y']).loss.backward()
        assert torch.isfinite(prediction).all()
    finally:
        context.tracker.close()


def test_unseen_test_prefix_distance_does_not_expand_training_slots(tmp_path):
    folder = strnn_fixture(tmp_path/'raw')
    original = make_context(configuration(tmp_path,'original-bounds'))
    try:
        expected = original.bundle.metadata.info['dis_slot_max']
    finally:
        original.tracker.close()
    geo = pd.read_csv(folder/'TaskTiny.geo')
    extra = geo.iloc[0].copy()
    extra['geo_id'],extra['coordinates'] = 999,'[0, 0]'
    geo.loc[len(geo)] = extra
    geo.to_csv(folder/'TaskTiny.geo',index=False)
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    last_trip = frame[frame.traj_id==frame.traj_id.max()]
    frame.loc[last_trip.index[0],'location'] = 999
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    changed = make_context(configuration(tmp_path,'changed-bounds'))
    try:
        assert changed.bundle.metadata.info['dis_slot_max']==expected
        assert max(float(row['dis'].max()) for row in changed.bundle.test.dataset.rows)>1000
        assert len(changed.model.dis_slots)==min(5,expected+1)
        batch = next(iter(changed.bundle.test))
        assert torch.isfinite(changed.model(changed.loss_fn.inputs(batch))).all()
    finally:
        changed.tracker.close()


def test_invalid_coordinates_fail_at_data_entry(tmp_path):
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
