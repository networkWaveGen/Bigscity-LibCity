"""Grid migration checks corrected axes against old math and complete real runs."""
import importlib
import itertools
import json

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import grid_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,name,experiment):
    parameters = dict(nb_residual_unit=1,filters=4,batch_norm=True) if name=='STResNet' else dict(
        res_repetation=1,res_nbfilter=4,lstm_channels=4,first_extnn_dropout=.1,lstm_dropout=.1)
    return dict(data=dict(dataset='GridTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),pad_training=False,
            sampling=dict(closeness=2,period=1,trend=1),
            features=dict(load_external=True,time_in_day=True,day_in_week=True,known_future_columns=['holiday'])),
        model=dict(name=name,params=parameters),training=dict(epochs=3,batch_size=16,
            scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        evaluation=dict(batch_size=17),runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


def canonical(source,metadata):
    rows,columns = metadata.input_layout['grid_shape']
    # Oracle derives positions directly from the atomic .geo, not model helpers.
    result = source.new_empty(len(source),source.shape[1],rows,columns,source.shape[-1])
    for index,position in enumerate(metadata.input_layout['node_positions']):
        result[:,:,int(position)//columns,int(position)%columns] = source[:,:,index]
    return result


def corrected_legacy(old,grid,batch,name):
    size,steps,rows,columns,features = grid.shape
    packed = grid.permute(0,1,4,2,3).reshape(size,steps*features,rows,columns)
    if name=='STResNet':
        prediction,start = 0,0
        for branch,count in zip(('c_way','p_way','t_way'),(old.len_closeness,old.len_period,old.len_trend)):
            if count:
                prediction = prediction+getattr(old,branch)(packed[:,start*features:(start+count)*features])
            start += count
        if old.ext_dim:
            prediction = prediction+old.external_ops(batch['future_features'][:,0]).reshape(size,old.output_dim,rows,columns)
    else:
        hidden = old.resnn(packed)
        if old.ext_dim:
            hidden = hidden+old.extnn(batch['history_features'])
        states = old.conv_lstm(hidden)
        count = old.lstm_channels*old.len_local
        local = hidden[:,:count]
        pred_c = old.conv_lstm_c(local*(1+old.concat_conv_c(local,states[:,:count])))
        global_ = hidden[:,count:]
        pred_t = old.conv_lstm_t(global_*(1+old.concat_conv_t(global_,states[:,count:])))
        if old.ext_dim:
            gate = torch.sigmoid(old.time_aware_extnn(batch['future_features'][:,0])).reshape(size,1,1,1)
            prediction = gate*pred_c+(1-gate)*pred_t
        else:
            prediction = (pred_c+pred_t)/2
    # Correct the legacy output view as well: channel is not the final grid axis.
    result = torch.tanh(prediction).permute(0,2,3,1).reshape(size,1,rows*columns,old.output_dim)
    positions = torch.tensor(np.array(old._test_positions,copy=True))
    return result[:,:,positions]


CASES = [('STResNet',bn,external,'split') for bn,external in itertools.product((False,True),repeat=2)]
CASES += [('ACFM',False,external,mode) for external in (False,True) for mode in ('split','split-chans','concat','cpt','cpt-sameoutput')]


@pytest.mark.parametrize('name,bn,external,mode',CASES)
def test_original_math_with_explicit_input_output_axis_corrections_and_all_gradients(tmp_path,name,bn,external,mode):
    grid_fixture(tmp_path/'raw',features=2)
    config = configuration(tmp_path,name,'reference')
    counts = (1,1,1) if mode=='cpt-sameoutput' else (2,1,1)
    config['data']['sampling'] = dict(zip(('closeness','period','trend'),counts))
    config['data']['features'] = dict(load_external=False,time_in_day=external,day_in_week=external)
    if name=='STResNet':
        config['model']['params'].update(filters=64,batch_norm=bn)
    else:
        config['model']['params'].update(res_bn=bn,lstm_channels=2,res_split_mode=mode,first_extnn_dropout=0.,lstm_dropout=0.)
    context = make_context(config)
    try:
        current = context.model.eval()
        meta = context.bundle.metadata
        module = importlib.import_module('libcity.model.traffic_flow_prediction.'+name)
        old = getattr(module,name)(dict(context.config['model']['params'],device=torch.device('cpu')),
            dict(feature_dim=2,output_dim=2,ext_dim=8 if external else 0,num_nodes=6,len_row=2,len_column=3,
                 len_closeness=counts[0],len_period=counts[1],len_trend=counts[2],scaler=NoneScaler())).eval()
        old._test_positions = meta.input_layout['node_positions']
        incoming = {key:value for key,value in old.state_dict().items() if key in current.state_dict()}
        loaded = current.load_state_dict(incoming,strict=False)
        assert set(loaded.missing_keys)=={'grid_order','node_positions'} and not loaded.unexpected_keys
        batch = {key:value[:2] for key,value in next(iter(context.bundle.train)).items()}
        grid = canonical(batch['X'],meta)
        expected = corrected_legacy(old,grid,batch,name)
        actual = current(model_inputs(batch,2))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        parameters = dict(old.named_parameters())
        for key,value in current.named_parameters():
            assert value.grad is not None
            torch.testing.assert_close(value.grad,parameters[key].grad,rtol=2e-4,atol=2e-6)
        assert all(not any(hasattr(part,key) for key in ('device','_device','_scaler','_logger')) for part in current.modules())
    finally:
        context.tracker.close()


def equal(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        for a,b in zip(left,right,strict=True):
            equal(a,b)
    else:
        assert left==right


@pytest.mark.parametrize('name',['STResNet','ACFM'])
def test_full_train_evaluate_and_exact_bn_dropout_optimizer_rng_resume(tmp_path,name):
    grid_fixture(tmp_path/'raw')
    full = run_experiment('train',config=configuration(tmp_path,name,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,name,'evaluated'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    info = json.loads(full.artifacts['runtime_info'].read_text())
    assert info['benchmark_protocol']=='grid_cpt' and info['split']['offsets']==[-1,-2,-24,-168]
    partial = make_context(configuration(tmp_path,name,'partial'))
    initial = {key:value.detach().clone() for key,value in partial.model.named_parameters()}
    train_loop(partial,until_epoch=1)
    assert any(not torch.equal(value,initial[key]) for key,value in partial.model.named_parameters())
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,name,'resumed'),resume=partial.directory/'checkpoints/last.pt')
    before,after = (torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','early_stopper','train_state','rng','loader_rng','best_model'):
        equal(before[key],after[key])
    export = np.load(evaluated.artifacts['predictions'],allow_pickle=False)
    assert export['prediction'].shape[1:]==(1,6,2)
    assert len(export['prediction'])==info['split']['test']


def test_atomic_row_order_cpt_timestamps_target_splits_scaler_and_cache_identity(tmp_path):
    folder = grid_fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'STResNet','data')
    first = make_context(cfg)
    try:
        bundle = first.bundle
        assert bundle.split_summary['cache']['status']=='miss'
        dataset = bundle.train.dataset
        assert dataset.offsets.tolist()==[-1,-2,-24,-168]
        raw = pd.read_csv(folder/'GridTiny.grid')
        ids = pd.read_csv(folder/'GridTiny.geo')
        batch = dataset[0]
        start = dataset.target_starts[0]
        for node,cell in ids.iterrows():
            subset = raw[(raw.row_id==cell.row_id)&(raw.column_id==cell.column_id)].sort_values('time')
            np.testing.assert_array_equal(batch['y_raw'][:,node].numpy(),subset.iloc[start:start+1][['value_0','value_1']].to_numpy(dtype=np.float32))
        targets = []
        for loader in (bundle.train,bundle.val,bundle.test):
            targets.append({int(start)+i for start in loader.dataset.target_starts for i in range(loader.dataset.horizon)})
        assert targets[0].isdisjoint(targets[1]) and targets[0].isdisjoint(targets[2]) and targets[1].isdisjoint(targets[2])
        assert bundle.metadata.adjacency is None
        assert bundle.metadata.input_layout['historical_external_dim']==10
        assert bundle.metadata.input_layout['future_external_dim']==9
        assert 'temperature' not in bundle.metadata.input_layout['future_feature_names']
        second = make_context(configuration(tmp_path,'STResNet','cache-hit'))
        try:
            assert second.bundle.split_summary['cache']['status']=='hit'
            torch.testing.assert_close(second.bundle.train.dataset[0]['X'],batch['X'],rtol=0,atol=0)
        finally:
            second.tracker.close()
    finally:
        first.tracker.close()


@pytest.mark.parametrize('name',['STResNet','ACFM'])
def test_target_and_unavailable_future_external_changes_do_not_change_inputs_or_forecast(tmp_path,name):
    folder = grid_fixture(tmp_path/'raw')
    first = make_context(configuration(tmp_path,name,'causal'))
    try:
        dataset = first.bundle.test.dataset
        before = dataset[0]
        mean = first.bundle.scaler.min
        start = int(dataset.target_starts[0])
        clock = pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(hours=start)
        timestamp = clock.strftime('%Y-%m-%dT%H:%M:%SZ')
        raw = pd.read_csv(folder/'GridTiny.grid')
        raw.loc[raw.time==timestamp,['value_0','value_1']] = 1e6
        raw.to_csv(folder/'GridTiny.grid',index=False)
        ext = pd.read_csv(folder/'GridTiny.ext')
        ext.loc[ext.time==timestamp,'temperature'] = 1e9
        ext.to_csv(folder/'GridTiny.ext',index=False)
        second = make_context(configuration(tmp_path,name,'changed'))
        try:
            after = second.bundle.test.dataset[0]
            assert second.bundle.split_summary['cache']['status']=='miss'
            assert second.bundle.scaler.min==mean
            for key in ('X','history_features','future_features'):
                torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
            assert not torch.equal(before['y_raw'],after['y_raw'])
            a = {key:value.unsqueeze(0) for key,value in before.items()}
            b = {key:value.unsqueeze(0) for key,value in after.items()}
            first.model.eval()
            torch.testing.assert_close(first.model(model_inputs(a,2)),first.model(model_inputs(b,2)),rtol=0,atol=0)
        finally:
            second.tracker.close()
    finally:
        first.tracker.close()


@pytest.mark.parametrize('name',['STResNet','ACFM'])
@pytest.mark.parametrize('counts',[(2,0,0),(0,1,1),(2,1,1)])
def test_variable_dimensions_single_branches_multistep_and_label_free_float64(tmp_path,name,counts):
    grid_fixture(tmp_path/'raw',features=4,outputs=3)
    cfg = configuration(tmp_path,name,'dimensions')
    cfg['data'].update(output_window=3,sampling=dict(zip(('closeness','period','trend'),counts)))
    context = make_context(cfg)
    try:
        batch = {key:value[:1] for key,value in next(iter(context.bundle.test)).items()}
        inputs = {key:value.double() for key,value in model_inputs(batch,3).items()}
        model = context.model.eval().double()
        source = inputs['X'].clone()
        state = {key:value.clone() for key,value in model.state_dict().items()}
        predicted = model(inputs)
        assert predicted.shape==(1,3,6,3) and predicted.dtype==torch.float64
        torch.testing.assert_close(model({**inputs,'y':torch.full_like(predicted,float('nan'))}),predicted,rtol=0,atol=0)
        torch.testing.assert_close(inputs['X'],source,rtol=0,atol=0)
        equal(state,model.state_dict())
        predicted.sum().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name',['STResNet','ACFM'])
def test_actual_singleton_grid_tail_batchnorm_cpu_amp(tmp_path,name):
    grid_fixture(tmp_path/'raw',steps=40,rows=1,columns=1)
    cfg = configuration(tmp_path,name,'amp')
    cfg['data']['sampling'] = dict(closeness=1,period=0,trend=0)
    cfg['training'].update(epochs=1,batch_size=11,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()
    state = torch.load(result.artifacts['last'],weights_only=True)
    assert state['train_state']['optimizer_steps']==3


def test_dyna_entity_id_fallback_and_missing_input_value_raw_target_separation(tmp_path):
    folder = grid_fixture(tmp_path/'raw')
    frame = pd.read_csv(folder/'GridTiny.grid')
    frame['entity_id'] = 100+frame.row_id*3+frame.column_id
    frame = frame.drop(columns=['row_id','column_id'])
    frame.loc[0,'value_0'] = np.nan
    frame.to_csv(folder/'GridTiny.dyna',index=False)
    (folder/'GridTiny.grid').unlink()
    context = make_context(configuration(tmp_path,'STResNet','dyna'))
    try:
        assert np.isnan(context.bundle.train.dataset.arrays['history']).any()
        assert torch.isfinite(context.bundle.train.dataset[0]['X']).all()
    finally:
        context.tracker.close()


def test_grid_source_file_change_is_rejected_before_resume(tmp_path):
    folder = grid_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'STResNet','partial'))
    train_loop(context,until_epoch=1)
    context.tracker.close()
    assert any(record['name']=='GridTiny.grid' for record in context.atomic_sources)
    raw = pd.read_csv(folder/'GridTiny.grid')
    latest = raw.time.max()
    raw.loc[raw.time==latest,'value_0'] += .5
    raw.to_csv(folder/'GridTiny.grid',index=False)
    with pytest.raises(ValueError,match='atomic source files changed'):
        run_experiment('train',config=configuration(tmp_path,'STResNet','rejected'),resume=context.directory/'checkpoints/last.pt')


@pytest.mark.parametrize('problem',['duplicate-cell','missing-observation','duplicate-observation','unknown-future-column'])
def test_invalid_grid_geometry_or_observations_are_rejected(tmp_path,problem):
    folder = grid_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'STResNet','invalid')
    if problem=='duplicate-cell':
        geo = pd.read_csv(folder/'GridTiny.geo')
        geo.loc[0,['row_id','column_id']] = geo.loc[1,['row_id','column_id']]
        geo.to_csv(folder/'GridTiny.geo',index=False)
    elif problem=='unknown-future-column':
        config['data']['features']['known_future_columns'] = ['value_0']
    else:
        raw = pd.read_csv(folder/'GridTiny.grid')
        raw = raw.iloc[1:] if problem=='missing-observation' else pd.concat([raw,raw.iloc[:1]])
        raw.to_csv(folder/'GridTiny.grid',index=False)
    with pytest.raises(ValueError,match='Grid|known_future_columns'):
        make_context(config)


def test_double_precision_raw_forecast_loss_does_not_round_small_residuals():
    from libcity.executor.evaluate import ForecastLoss
    loss = ForecastLoss(dict(name='masked_mse',mask_value=0.),NoneScaler(),1)
    prediction = torch.tensor([[[[1.+1e-10]]]],dtype=torch.float64,requires_grad=True)
    target = torch.ones_like(prediction)
    result = loss(prediction,target,target)
    assert result.loss.dtype==torch.float64 and result.loss>0
    result.loss.backward()
    torch.testing.assert_close(prediction.grad,torch.full_like(prediction,2e-10),rtol=0,atol=2e-17)


@pytest.mark.parametrize('boundary',['gap','separate-files'])
def test_history_never_crosses_missing_timestamps_or_atomic_file_boundaries(tmp_path,boundary):
    folder = grid_fixture(tmp_path/'raw',steps=700)
    frame = pd.read_csv(folder/'GridTiny.grid')
    metadata_path = folder/'config.json'
    if boundary=='gap':
        gap = '2024-01-01T20:00:00Z'
        frame[frame.time!=gap].to_csv(folder/'GridTiny.grid',index=False)
    else:
        midpoint = '2024-01-15T14:00:00Z'
        frame[frame.time<midpoint].to_csv(folder/'first.grid',index=False)
        frame[frame.time>=midpoint].to_csv(folder/'second.grid',index=False)
        raw = json.loads(metadata_path.read_text())
        raw['data_files'] = ['first','second']
        metadata_path.write_text(json.dumps(raw))
    context = make_context(configuration(tmp_path,'ACFM','boundaries'))
    try:
        for loader in (context.bundle.train,context.bundle.val,context.bundle.test):
            dataset = loader.dataset
            clocks = dataset.arrays['clock']
            for start in dataset.target_starts:
                left = start+dataset.offsets.min()
                assert np.all(np.diff(clocks[left:start+dataset.horizon])==np.timedelta64(3600,'s'))
                if boundary=='separate-files':
                    assert start<350 or left>=350
    finally:
        context.tracker.close()


@pytest.mark.parametrize('damage',['missing','shape'])
def test_shared_grid_cache_handles_horizon_changes_and_missing_fields(tmp_path,damage):
    grid_fixture(tmp_path/'raw')
    initial = make_context(configuration(tmp_path,'STResNet','first'))
    path = initial.bundle.split_summary['cache']['path']
    count = len(initial.bundle.train.dataset)
    initial.tracker.close()
    cfg = configuration(tmp_path,'STResNet','horizon')
    cfg['data']['output_window'] = 2
    changed = make_context(cfg)
    try:
        assert changed.bundle.split_summary['cache']['status']=='hit'
        assert changed.bundle.split_summary['cache']['path']==path
        assert len(changed.bundle.train.dataset)==count-1
        with np.load(path,allow_pickle=False) as archive:
            arrays = {name:archive[name] for name in archive.files if name!='known' or damage=='shape'}
        if damage=='shape':
            arrays['known'] = arrays['known'][:-1]
        np.savez_compressed(path,**arrays)
        rebuilt = make_context(configuration(tmp_path,'STResNet','rebuilt'))
        try:
            assert rebuilt.bundle.split_summary['cache']['status']=='miss'
            assert rebuilt.bundle.train.dataset[0]['future_features'].shape==(1,9)
        finally:
            rebuilt.tracker.close()
    finally:
        changed.tracker.close()


@pytest.mark.parametrize('params',[dict(res_split_mode='none'),dict(res_split_mode='split-chans',lstm_channels=4),
                                 dict(res_split_mode='cpt-sameoutput')])
def test_incompatible_residual_modes_fail_with_specific_error(tmp_path,params):
    grid_fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,'ACFM','invalid-residual')
    cfg['model']['params'].update(params)
    with pytest.raises(ValueError,match='ACFM'):
        make_context(cfg)


@pytest.mark.parametrize('name,sampling',[('STResNet','window'),('GRU','grid_cpt')])
def test_grid_sampling_and_model_must_be_selected_together(name,sampling):
    with pytest.raises(ValueError,match='grid_cpt'):
        load_config(config=dict(data=dict(dataset='unused',sampling=dict(name=sampling)),model=dict(name=name)))
