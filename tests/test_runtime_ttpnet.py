"""TTPNet's native route contract, cumulative objective and complete lifecycle."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.data.dataset.eta_encoder.ttpnet_encoder import TtpnetEncoder
from libcity.data.task_bundles import collate_sequences,prepare_eta
from libcity.executor.train import train_loop
from libcity.executor.ttpnet_step import cumulative_step
from libcity.model.eta.TTPNet import TTPNet as ReferenceTTPNet
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import ttpnet_fixture
from runtime_helpers import make_context


def configuration(root,name):
    return dict(data=dict(dataset='TaskTiny',data_dir=str(root/'raw'),cache=False),
        model=dict(name='TTPNet',params=dict(uid_emb_size=4,weekid_emb_size=2,timid_emb_size=3)),
        training=dict(epochs=2,batch_size=4,shuffle=True,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


def raw_trip(frame,identity):
    first = frame.loc[frame.dyna_id==identity].iloc[0]
    return frame[(frame.entity_id==first.entity_id)&(frame.traj_id==first.traj_id)].sort_values('time')


def original_batch(frame,rows,metadata):
    trips = [raw_trip(frame,int(row['sample_id'])) for row in rows]
    points = [np.asarray([json.loads(value) for value in trip.coordinates]) for trip in trips]
    values = dict(current_longi=torch.tensor(np.stack([coords[:,0] for coords in points]),dtype=torch.float64),
        current_lati=torch.tensor(np.stack([coords[:,1] for coords in points]),dtype=torch.float64),
        current_loc=torch.stack([row['road_id'] for row in rows]),current_dis=torch.tensor(np.stack([trip.current_dis for trip in trips]),dtype=torch.float64))
    for name in ('speeds','speeds_relevant1','speeds_relevant2','speeds_long'):
        values[name] = torch.tensor(np.stack([np.concatenate([json.loads(value) for value in trip[name]]) for trip in trips]),dtype=torch.float64)
    values['grid_len'] = torch.tensor(np.stack([trip.grid_len for trip in trips]),dtype=torch.float64)
    values['dist'] = torch.tensor([[float(trip.current_dis.iloc[-1]-trip.current_dis.iloc[0])] for trip in trips],dtype=torch.float64)
    for name in ('uid','weekid','timeid','holiday'):
        values[name] = torch.stack([row[name].reshape(1) for row in rows])
    return values


def test_original_speed_road_bidirectional_cumulative_predictions_and_all_gradients(tmp_path):
    folder = ttpnet_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'reference'))
    try:
        model = context.model.double().eval()
        metadata = context.bundle.metadata
        feature = {f'{name}_{suffix}':value for name,pair in metadata.info['statistics'].items()
            for suffix,value in zip(('mean','std'),pair)}
        feature['geo_embedding'] = np.zeros((128*128,32),dtype=np.float32)
        feature['geo_embedding'][:8] = metadata.info['road_embeddings']
        feature['uid_size'] = 2
        reference = ReferenceTTPNet(dict(uid_emb_size=4,weekid_emb_size=2,timdid_emb_size=3),feature).double().eval()
        weights = reference.state_dict()
        for name,value in model.state_dict().items(): weights[name] = weights[name][:value.shape[0]]
        model.load_state_dict({name:weights[name] for name in model.state_dict()})
        rows = [row for row in context.bundle.train.dataset.rows if int(row['lengths'])==3][:2]
        frame = pd.read_csv(folder/'TaskTiny.dyna')
        legacy = original_batch(frame,rows,metadata)
        batch = {name:value.double() if value.is_floating_point() else value for name,value in collate_sequences(rows).items()}
        old = reference(legacy)
        actual,cumulative,valid = model.forward_with_aux(context.loss_fn.inputs(batch))
        torch.testing.assert_close(cumulative,old,rtol=3e-5,atol=2e-6)
        torch.testing.assert_close(actual,old[:,-1:],rtol=3e-5,atol=2e-6)
        expected_raw = old[:,-1:]*feature['time_gap_std']+feature['time_gap_mean']
        torch.testing.assert_close(context.bundle.scaler.inverse_transform(actual),expected_raw)
        assert valid.all()
        old.square().mean().backward();cumulative.square().mean().backward()
        parameters = dict(reference.named_parameters())
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,parameters[name].grad[:parameter.shape[0]],rtol=5e-4,atol=2e-6)
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','data_feature','_logger')) for module in model.modules())
        embedding = model.bi_lstm.road_lstm.Road_net.embedding
        assert embedding.num_embeddings==8 and embedding.weight.requires_grad
    finally: context.tracker.close()


def test_cumulative_relative_rmse_matches_route_weighting_gradients_and_exact_fit(tmp_path):
    ttpnet_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'loss'))
    try:
        mean,std = context.bundle.metadata.info['statistics']['time_gap']
        target = torch.tensor([[0.,60.,120.,float('nan'),float('nan')],[0.,70.,140.,210.,280.]],dtype=torch.float64)
        valid = torch.tensor([[True,True,False,False],[True,True,True,True]])
        seconds = torch.tensor([[65.,110.,999.,999.],[60.,150.,200.,300.]],dtype=torch.float64)
        normalized = ((seconds-mean)/std).requires_grad_()
        actual = context.loss_fn.cumulative(normalized,target,valid)
        expected = torch.stack([(((seconds[row,mask]-target[row,1:][mask])/target[row,1:][mask]).square().mean()).sqrt()
            for row,mask in enumerate(valid)]).mean()
        torch.testing.assert_close(actual.loss,expected,rtol=1e-12,atol=1e-12)
        reference = normalized*std+mean
        reference = torch.stack([(((reference[row,mask]-target[row,1:][mask])/target[row,1:][mask]).square().mean()).sqrt()
            for row,mask in enumerate(valid)]).mean()
        torch.testing.assert_close(torch.autograd.grad(actual.loss,normalized,retain_graph=True)[0],torch.autograd.grad(reference,normalized)[0])
        exact = ((target[:,1:].nan_to_num(0.)-mean)/std).requires_grad_()
        zero = context.loss_fn.cumulative(exact,target,valid).loss
        zero.backward()
        assert zero.item()==0. and torch.isfinite(exact.grad).all() and not exact.grad.any()
        raw = torch.tensor([[200.],[300.]],dtype=torch.float64)
        prediction = (raw+20-mean)/std
        result = context.loss_fn(prediction,(raw-mean)/std,raw)
        torch.testing.assert_close(result.loss,(20/raw).mean(),rtol=1e-12,atol=1e-12)
        assert context.loss_fn.raw(prediction,(raw-mean)/std,raw)[0].dtype==torch.float64
    finally: context.tracker.close()


def test_complete_training_seconds_export_and_exact_dropout_resume(tmp_path):
    ttpnet_fixture(tmp_path/'raw')
    full = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluation = run_experiment('evaluate',config=configuration(tmp_path,'evaluation'),checkpoint=full.artifacts['best'])
    assert evaluation.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(configuration(tmp_path,'partial'))
    train_loop(partial,cumulative_step(partial.loss_fn),until_epoch=1);partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert actual['train_state']==expected['train_state'] and actual['scheduler']==expected['scheduler']
    torch.testing.assert_close(actual['rng']['torch'],expected['rng']['torch'],rtol=0,atol=0)
    for key,value in expected['model'].items(): torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)
    records = json.loads(evaluation.artifacts['predictions'].read_text())
    assert records and all(set(row)=={'sample_id','traj_id','prediction_seconds','truth_seconds'} for row in records)
    info = json.loads(evaluation.artifacts['runtime_info'].read_text())
    assert info['benchmark_protocol']=='ttpnet_eta' and info['split']['duration_unit']=='seconds'


def test_variable_padding_batch_one_float64_and_label_free_input_ownership(tmp_path):
    ttpnet_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'padding'))
    try:
        model = context.model.double().eval()
        inputs = context.loss_fn.inputs(next(iter(context.bundle.train)))
        inputs = {name:value.double() if value.is_floating_point() else value for name,value in inputs.items()}
        saved = {name:value.clone() for name,value in inputs.items()}
        state = {name:value.clone() for name,value in model.state_dict().items()}
        rng = torch.get_rng_state()
        prediction = model(inputs)
        assert prediction.shape==(len(inputs['lengths']),1) and prediction.dtype==torch.float64 and torch.isfinite(prediction).all()
        for index,length in enumerate(inputs['lengths']):
            one = {name:value[index:index+1,:int(length)] if name in ('coords','road_id','short_traffic','long_traffic')
                else value[index:index+1] for name,value in inputs.items()}
            torch.testing.assert_close(model(one),prediction[index:index+1],rtol=1e-10,atol=1e-10)
        poisoned = {name:value.clone() for name,value in inputs.items()}
        for name in ('coords','road_id','short_traffic','long_traffic'):
            values = poisoned[name]
            valid = (torch.arange(values.shape[1])[None]<inputs['lengths'][:,None]).reshape(*values.shape[:2],*([1]*(values.ndim-2)))
            fill = float('nan') if values.is_floating_point() else 99999
            values = values.masked_fill(~valid,fill)
            poisoned[name] = torch.cat((values,values.new_full((len(values),3,*values.shape[2:]),fill)),1)
        poisoned.update(y=torch.tensor(float('nan')),current_tim=torch.tensor(float('nan')),local_y_raw=torch.tensor(float('nan')))
        torch.testing.assert_close(model(poisoned),prediction,rtol=0,atol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for name,value in saved.items(): torch.testing.assert_close(value,inputs[name],rtol=0,atol=0)
        for name,value in state.items(): torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
        prediction.square().mean().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        with pytest.raises(ValueError,match='at least two'): model({**inputs,'lengths':torch.ones_like(inputs['lengths'])})
    finally: context.tracker.close()


def test_train_only_statistics_default_benchmark_samples_units_and_static_ids(tmp_path):
    folder = ttpnet_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'features'))
    try:
        cfg = context.config
        base = prepare_eta(cfg['data'],cfg['training'],cfg['evaluation'],17)
        for split in ('train','val','test'):
            assert [int(row['sample_id']) for row in getattr(context.bundle,split).dataset.rows]==[
                int(row['sample_id']) for row in getattr(base,split).dataset.rows]
        frame = pd.read_csv(folder/'TaskTiny.dyna')
        train = [raw_trip(frame,int(row['sample_id'])) for row in context.bundle.train.dataset.rows]
        expected = np.concatenate([np.concatenate([json.loads(value) for value in trip.speeds]) for trip in train])
        np.testing.assert_allclose(context.bundle.metadata.info['statistics']['speeds'],[expected.mean(),expected.std()])
        ids = context.bundle.metadata.info['road_ids']
        for row in context.bundle.train.dataset.rows:
            trip = raw_trip(frame,int(row['sample_id']))
            assert [ids[index] for index in row['road_id']]==trip.location.tolist()
            assert row['holiday'].item()==int(pd.Timestamp(trip.time.iloc[0]).day_of_week>=5)
        saved = [{key:value.clone() for key,value in row.items()} for row in context.bundle.train.dataset.rows]
        for name in ('speeds','speeds_relevant1','speeds_relevant2','speeds_long'):
            frame[name] = frame[name].map(lambda value:json.dumps((np.asarray(json.loads(value))/3.6).tolist()))
        frame['grid_len'] *= 1000;frame['current_dis'] *= 1000
        frame.to_csv(folder/'TaskTiny.dyna',index=False)
        raw = json.loads((folder/'config.json').read_text());raw.update(speed_unit='m/s',grid_len_unit='m',distance_unit='m')
        (folder/'config.json').write_text(json.dumps(raw),encoding='utf-8')
        converted = make_context(configuration(tmp_path,'converted'))
        try:
            for old,new in zip(saved,converted.bundle.train.dataset.rows):
                for name,value in old.items(): torch.testing.assert_close(value,new[name],rtol=1e-5,atol=2e-6)
        finally: converted.tracker.close()
    finally: context.tracker.close()


def test_actual_future_elapsed_targets_do_not_change_departure_inputs(tmp_path):
    folder = ttpnet_fixture(tmp_path/'raw')
    before = make_context(configuration(tmp_path,'before'))
    try:
        row = before.bundle.test.dataset.rows[-1]
        saved = {name:value.clone() for name,value in before.loss_fn.inputs(row).items()}
        prediction = before.model.eval()(before.loss_fn.inputs(collate_sequences([row])))
        frame = pd.read_csv(folder/'TaskTiny.dyna')
        trip = raw_trip(frame,int(row['sample_id']))
        for index in trip.index[1:]:
            frame.loc[index,'time'] = (pd.Timestamp(frame.loc[index,'time'])+pd.Timedelta(seconds=90)).strftime('%Y-%m-%dT%H:%M:%SZ')
        frame.to_csv(folder/'TaskTiny.dyna',index=False)
        after = make_context(configuration(tmp_path,'after'))
        try:
            new = next(value for value in after.bundle.test.dataset.rows if int(value['sample_id'])==int(row['sample_id']))
            assert new['y_raw'].item()!=row['y_raw'].item()
            for name,value in saved.items(): torch.testing.assert_close(value,new[name],rtol=0,atol=0)
            assert before.bundle.metadata.info['statistics']==after.bundle.metadata.info['statistics']
            torch.testing.assert_close(prediction,after.model.eval()(after.loss_fn.inputs(collate_sequences([new]))),rtol=0,atol=0)
        finally: after.tracker.close()
    finally: before.tracker.close()


@pytest.mark.parametrize('problem',['missing_snapshot','future_snapshot','speed_width','negative_speed','bad_json','unknown_road','bad_embedding'])
def test_invalid_or_future_inputs_fail_before_training(tmp_path,problem):
    folder = ttpnet_fixture(tmp_path/'raw')
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    if problem=='missing_snapshot': frame = frame.drop(columns='traffic_snapshot_time')
    elif problem=='future_snapshot': frame.loc[1,'traffic_snapshot_time'] = '2024-01-01T00:01:00Z'
    elif problem=='speed_width': frame.loc[1,'speeds'] = '[1,2,3]'
    elif problem=='negative_speed': frame.loc[1,'speeds'] = '[-1,2,3,4]'
    elif problem=='bad_json': frame.loc[1,'speeds'] = '__import__("os").getcwd()'
    elif problem=='unknown_road': frame.loc[1,'location'] = 9999
    else:
        geo = pd.read_csv(folder/'TaskTiny.geo');geo.loc[0,'embeddings'] = '[1,2]';geo.to_csv(folder/'TaskTiny.geo',index=False)
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    with pytest.raises(ValueError): make_context(configuration(tmp_path,'invalid'))


def test_actual_cpu_amp_with_zero_speeds_and_constant_features(tmp_path):
    folder = ttpnet_fixture(tmp_path/'raw')
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    for name,width in (('speeds',4),('speeds_relevant1',4),('speeds_relevant2',4),('speeds_long',7)):
        frame[name] = json.dumps([0.]*width)
    frame['coordinates'] = '[116,39]';frame['current_dis'] = 0.;frame['grid_len'] = .1
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    cfg = configuration(tmp_path,'amp');cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()
    last = torch.load(result.artifacts['last'],weights_only=True)
    assert last['train_state']['optimizer_steps']>0


def test_two_point_routes_and_nonfixed_road_embedding_width(tmp_path):
    folder = ttpnet_fixture(tmp_path/'raw')
    frame = pd.read_csv(folder/'TaskTiny.dyna').groupby(['entity_id','traj_id'],sort=False).head(2)
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    geo = pd.read_csv(folder/'TaskTiny.geo');geo['embeddings'] = geo.embeddings.map(lambda value:json.dumps(json.loads(value)[:5]))
    geo.to_csv(folder/'TaskTiny.geo',index=False)
    cfg = configuration(tmp_path,'two');cfg['data']['min_route_length'] = 2;cfg['training']['epochs'] = 1
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()
    weights = torch.load(result.artifacts['best'],weights_only=True)
    assert weights['bi_lstm.road_lstm.Road_net.embedding.weight'].shape==(8,5)


def test_multiday_seconds_and_legacy_safe_encoder_calendar(tmp_path,monkeypatch):
    folder = ttpnet_fixture(tmp_path/'raw_data')
    monkeypatch.chdir(tmp_path)
    encoder = TtpnetEncoder(dict(dataset='TaskTiny',eta_encoder='TtpnetEncoder'))
    assert encoder.traj_len_idx==list(encoder.feature_dict).index('traj_len')
    frame = pd.read_csv(folder/'TaskTiny.dyna')
    saturday = frame.loc[frame.traj_id==5].copy()
    for position,index in enumerate(saturday.index):
        saturday.loc[index,'time'] = (pd.Timestamp('2024-01-06',tz='UTC')+pd.Timedelta(hours=position*25)).strftime('%Y-%m-%dT%H:%M:%SZ')
    encoded = encoder.encode(0,[saturday.values.tolist()],{name:index for index,name in enumerate(frame.columns)})[0]
    assert encoded[3][-1]==(len(saturday)-1)*25*3600 and encoded[16][0]==encoded[3][-1] and encoded[15]==[1]
    monday = frame.loc[frame.traj_id==0]
    assert encoder.encode(0,[monday.values.tolist()],{name:index for index,name in enumerate(frame.columns)})[0][15]==[0]
    for trip_id,group in frame.groupby('traj_id'):
        for position,index in enumerate(group.index):
            frame.loc[index,'time'] = (pd.Timestamp('2024-02-01',tz='UTC')+pd.Timedelta(days=int(trip_id)*7,hours=position*25)).strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.to_csv(folder/'TaskTiny.dyna',index=False)
    cfg = configuration(tmp_path,'multiday');cfg['data']['data_dir'] = str(tmp_path/'raw_data')
    context = make_context(cfg)
    try:
        assert all(row['y_raw'].item()>86400 for row in context.bundle.train.dataset.rows)
        assert context.bundle.metadata.info['statistics']['time_gap'][0]==25*3600
    finally: context.tracker.close()
