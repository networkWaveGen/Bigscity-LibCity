"""OD atoms and all spatial metric axes are prerequisites for GEML/CSTN."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data.od_bundle import prepare_od
from libcity.data.sources import atomic_sources
from libcity.evaluator.runtime_metrics import ForecastMetrics
from libcity.executor.evaluate import ForecastLoss,evaluate_loop,model_inputs
from libcity.executor.train import build_context,train_loop
from libcity.pipeline.experiment import initialize_runtime
from libcity.testing.fixtures import od_fixture
from libcity.utils.normalization import StandardScaler

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,name='probe'):
    config = load_config(config=dict(model=dict(name='GRU',params=dict(hidden_size=4)),
        data=dict(dataset='ODTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),input_window=4,
            output_window=3,pad_training=False,graph=dict(name='weighted'),features=dict(load_external=True,
            time_in_day=True,day_in_week=True,known_future_columns=['holiday'])),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2),evaluation=dict(batch_size=4),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name)))
    # This is a component-level adapter test, not a GRU/GEML CLI registration.
    config['data']['sampling'] = dict(name='od_window')
    return config


def bundle(config):
    return prepare_od(config['data'],config['training'],config['evaluation'],config['runtime']['seed'])


@pytest.mark.parametrize('format',('od','dyna','gridod'))
def test_shuffled_od_and_grid_atoms_align_both_axes_to_geo_ids_and_keep_features_separate(tmp_path,format):
    grid = (2,2) if format=='gridod' else None
    folder = od_fixture(tmp_path/'raw',grid_shape=grid,extension='.'+format)
    result = bundle(configuration(tmp_path))
    nodes = 4 if grid else 3
    values = result.train.dataset.arrays['history']
    assert values.shape==(48,nodes,nodes,2)
    for origin in range(nodes):
        for destination in range(nodes):
            assert values[5,origin,destination,0]==pytest.approx(10.5+origin*3+destination*.7)
    ids = pd.read_csv(folder/'ODTiny.geo')['geo_id'].tolist()
    assert list(result.metadata.info['node_ids'])==ids
    assert result.metadata.adjacency[0,1]==2 and result.metadata.adjacency[1,0]==3
    assert not result.metadata.adjacency.flags.writeable
    assert result.metadata.input_layout['kind']=='od_window'
    assert result.metadata.input_layout['grid_shape']==(2,2) if grid else result.metadata.input_layout['grid_shape'] is None
    sample = result.train.dataset[0]
    assert sample['X'].shape==(4,nodes,nodes,2) and sample['y_raw'].shape==(3,nodes,nodes,1)
    assert sample['history_features'].shape==(4,10) and sample['future_features'].shape==(3,9)
    assert sample['history_mask'].dtype==torch.bool
    assert result.metadata.feature_dim==2  # Calendar/weather never become OD flow channels.
    assert list(result.metadata.input_layout['future_feature_names'])==[
        'time_of_day',*[f'day_of_week_{index}' for index in range(7)],'holiday']
    inputs = model_inputs(sample,1)
    assert 'history_mask' in inputs and 'y' not in inputs and 'y_raw' not in inputs
    records = atomic_sources(configuration(tmp_path)['data'])
    assert any(row['name']=='ODTiny.'+format and row['size'] for row in records)
    assert not any(row['name']=='ODTiny.dyna' and row['size'] is None for row in records)


def test_target_intervals_are_disjoint_scaler_is_train_only_and_raw_positive_mask_survives_scaling(tmp_path):
    folder = od_fixture(tmp_path/'raw')
    config = configuration(tmp_path)
    result = bundle(config)
    sets = []
    for loader in (result.train,result.val,result.test):
        sets.append({int(start+step) for start in loader.dataset.target_starts for step in range(3)})
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert result.val.dataset.target_starts[0]==28 and result.val.dataset.target_starts[0]-4<28
    raw = result.train.dataset.arrays['history']
    assert result.scaler.mean==pytest.approx(raw[:28,...,:1].mean())
    assert result.scaler.std==pytest.approx(raw[:28,...,:1].std())
    index = list(result.train.dataset.target_starts).index(7)
    sample = result.train.dataset[index]
    assert sample['X'][-1,1,2,0]!=0 and not sample['history_mask'][-1,1,2,0]
    path = folder/'ODTiny.od'
    frame = pd.read_csv(path)
    times = sorted(frame['time'].unique())
    frame.loc[frame['time'].isin(times[28:]),['flow_0','flow_1']] += 1000
    frame.to_csv(path,index=False)
    ext = pd.read_csv(folder/'ODTiny.ext')
    ext.loc[ext['time'].isin(times[28:]),'weather'] += 500
    ext.to_csv(folder/'ODTiny.ext',index=False)
    changed = bundle(config)
    assert vars(changed.scaler)==vars(result.scaler)
    for key in ('X','history_features','future_features','history_mask'):
        torch.testing.assert_close(changed.val.dataset[0][key],result.val.dataset[0][key],rtol=0,atol=0)
    assert not torch.equal(changed.val.dataset[0]['y_raw'],result.val.dataset[0]['y_raw'])


def test_missing_history_is_zero_imputed_but_raw_targets_and_masks_remain_exact(tmp_path):
    folder = od_fixture(tmp_path/'raw')
    path = folder/'ODTiny.od'
    frame = pd.read_csv(path)
    ids = pd.read_csv(folder/'ODTiny.geo')['geo_id'].tolist()
    row = (frame['time']=='2024-01-01T00:30:00Z')&(frame['origin_id']==ids[1])&(frame['destination_id']==ids[2])
    frame.loc[row,'flow_0'] = np.nan
    frame.to_csv(path,index=False)
    result = bundle(configuration(tmp_path))
    index = list(result.train.dataset.target_starts).index(7)
    sample = result.train.dataset[index]
    assert torch.isfinite(sample['X']).all() and not sample['history_mask'][-1,1,2,0]
    target_index = list(result.train.dataset.target_starts).index(6)
    assert torch.isnan(result.train.dataset[target_index]['y_raw'][0,1,2,0])


@pytest.mark.parametrize('damage',('shape','dtype','known'))
def test_raw_cache_reuses_windows_split_scaler_and_batch_changes_and_rebuilds_damage(tmp_path,monkeypatch,damage):
    folder = od_fixture(tmp_path/'raw')
    config = configuration(tmp_path)
    first = bundle(config)
    assert first.split_summary['cache']['status']=='miss'
    original = pd.read_csv
    def guarded(path,*args,**kwargs):
        if Path(path).suffix in ('.od','.ext'):
            raise AssertionError('Raw observations should come from the cache')
        return original(path,*args,**kwargs)
    config['data'].update(input_window=2,output_window=1,pad_training=True,scaler=dict(name='none'),split=dict(train=.5,val=.25))
    config['training']['batch_size'] = 7
    with monkeypatch.context() as scope:
        scope.setattr(pd,'read_csv',guarded)
        second = bundle(config)
    assert second.split_summary['cache']['status']=='hit' and len(second.train.dataset)%7==0
    assert len(second.val.dataset)%7!=0  # No validation padding.
    path = Path(first.split_summary['cache']['path'])
    with np.load(path,allow_pickle=False) as archive:
        arrays = {name:archive[name] for name in archive.files}
    if damage=='shape':
        arrays['history'] = arrays['history'][...,:1]
    elif damage=='dtype':
        arrays['history'] = np.full(arrays['history'].shape,'bad')
    else:
        arrays['known'][0,0] = np.inf
    np.savez_compressed(path,**arrays)
    assert bundle(config).split_summary['cache']['status']=='miss'
    relation = pd.read_csv(folder/'ODTiny.rel');relation['distance'] += 1
    relation.to_csv(folder/'ODTiny.rel',index=False)
    assert bundle(config).split_summary['cache']['status']=='miss'


def test_windows_do_not_cross_clock_gaps_or_data_file_boundaries(tmp_path):
    folder = od_fixture(tmp_path/'raw')
    path = folder/'ODTiny.od'
    frame = pd.read_csv(path)
    times = sorted(frame['time'].unique())
    first = frame[frame['time'].isin(times[:24])]
    second = frame[frame['time'].isin(times[24:])].copy()
    second['time'] = (pd.to_datetime(second['time'],utc=True)+pd.Timedelta(hours=2)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    first.to_csv(folder/'part_a.od',index=False);second.to_csv(folder/'part_b.od',index=False)
    external = pd.read_csv(folder/'ODTiny.ext')
    shifted = external['time'].isin(times[24:])
    external.loc[shifted,'time'] = (pd.to_datetime(external.loc[shifted,'time'],utc=True)+
        pd.Timedelta(hours=2)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    external.to_csv(folder/'ODTiny.ext',index=False)
    meta = json.loads((folder/'config.json').read_text());meta['data_files']=['part_a','part_b']
    (folder/'config.json').write_text(json.dumps(meta))
    result = bundle(configuration(tmp_path))
    for loader in (result.train,result.val,result.test):
        for start in loader.dataset.target_starts:
            assert not start-4<24<=start+2
    assert {row['name'] for row in atomic_sources(configuration(tmp_path)['data'])}>={'part_a.od','part_b.od'}


@pytest.mark.parametrize('problem',('duplicate','missing','unknown','infinite'))
def test_malformed_dense_od_rows_fail_clearly(tmp_path,problem):
    folder = od_fixture(tmp_path/'raw')
    path = folder/'ODTiny.od';frame = pd.read_csv(path)
    if problem=='duplicate':
        frame = pd.concat([frame,frame.iloc[:1]])
    elif problem=='missing':
        frame = frame.iloc[1:]
    elif problem=='unknown':
        frame.loc[0,'origin_id'] = 999999
    else:
        frame.loc[0,'flow_0'] = np.inf
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='OD'):
        bundle(configuration(tmp_path))


@pytest.mark.parametrize('spatial',((4,2),(2,3,1),(2,1,2,1,1)))
@pytest.mark.parametrize('mode',('single','average'))
def test_node_od_and_grid_od_metrics_match_direct_numpy_over_all_spatial_axes(spatial,mode):
    rng = np.random.default_rng(17)
    target = rng.normal(size=(5,3,*spatial))
    prediction = target+rng.normal(size=target.shape)*.3
    target.ravel()[0] = 0.;target.ravel()[3] = np.nan
    accumulator = ForecastMetrics(['MAE','MSE','RMSE','MAPE','R2','EVAR'],0.,mode)
    for left,right in ((0,2),(2,3),(3,5)):
        accumulator.update(torch.from_numpy(prediction[left:right]),torch.from_numpy(target[left:right]))
    actual = accumulator.compute()
    for horizon in range(3):
        indices = slice(0,horizon+1) if mode=='average' else slice(horizon,horizon+1)
        truth,forecast = target[:,indices].ravel(),prediction[:,indices].ravel()
        valid = np.isfinite(truth)&(truth!=0)
        truth,forecast = truth[valid],forecast[valid]
        error = forecast-truth
        expected = dict(MAE=np.abs(error).mean(),MSE=np.square(error).mean(),RMSE=np.sqrt(np.square(error).mean()),
            MAPE=np.abs(error/truth).mean(),R2=1-np.square(error).sum()/np.square(truth-truth.mean()).sum(),
            EVAR=1-error.var()/truth.var())
        for name,value in expected.items():
            assert actual[f'{name}@{horizon+1}']==pytest.approx(float(value),rel=2e-12,abs=2e-12)


class ODProbe(torch.nn.Module):
    def __init__(self,horizon):
        super().__init__()
        self.horizon = horizon
        self.scale = torch.nn.Parameter(torch.tensor(.5))
        self.bias = torch.nn.Parameter(torch.tensor(.1))

    def forward(self,inputs):
        assert 'y' not in inputs and inputs['history_mask'].dtype==torch.bool
        source = inputs['X'][:,-1:,...,:1]
        return (source*self.scale+self.bias).expand(-1,self.horizon,-1,-1,-1)


def equal(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True):
            equal(a,b)
    else:
        assert left==right


def test_real_od_generic_train_export_and_checkpoint_resume_without_claiming_native_model_migration(tmp_path):
    od_fixture(tmp_path/'raw')
    def context(name):
        config = configuration(tmp_path,name)
        config['model'] = dict(name='ODProbe',params=dict(horizon=3))
        initialize_runtime(config['runtime'])
        data = bundle(config)
        directory = tmp_path/'runs'/name;directory.mkdir(parents=True)
        model = ODProbe(3)
        return build_context(model,data,config,torch.device('cpu'),directory,logging.getLogger('od-probe'))
    full = context('full')
    contexts = [full]
    try:
        train_loop(full)
        evaluated = evaluate_loop(full.model,full.bundle.test,device=full.device,loss_fn=full.loss_fn,
            evaluation=full.config['evaluation'],directory=full.directory/'eval')
        assert np.isfinite(list(evaluated.metrics.values())).all()
        with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
            assert archive['prediction'].shape[1:]==(3,3,3,1)
        partial = context('partial');contexts.append(partial)
        train_loop(partial,until_epoch=1)
        resumed = context('resumed');contexts.append(resumed)
        resumed.checkpoints.load_last(partial.directory/'checkpoints/last.pt',resumed)
        train_loop(resumed)
        expected,actual = (torch.load(item.directory/'checkpoints/last.pt',weights_only=True) for item in (full,resumed))
        for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
            equal(expected[key],actual[key])
        source = tmp_path/'raw/ODTiny/ODTiny.od'
        frame = pd.read_csv(source)
        frame.loc[frame['time']==frame['time'].max(),'flow_0'] += 1
        frame.to_csv(source,index=False)
        changed = context('changed');contexts.append(changed)
        with pytest.raises(ValueError,match='source files changed'):
            changed.checkpoints.load_last(partial.directory/'checkpoints/last.pt',changed)
    finally:
        for item in contexts:
            item.tracker.close()
