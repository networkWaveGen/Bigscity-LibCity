"""Point observations follow real geo IDs and UTC time, never CSV row blocks."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data.traffic_bundle import prepare_traffic
from libcity.data.traffic_cache import FIELDS,prepare_arrays
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture

pytestmark=pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def fixture(root,text=False,order='shuffled'):
    folder=atomic_fixture(root/'raw')
    ids=['west','east','center'] if text else [71,3,108]
    path=folder/'Tiny.geo';geo=pd.read_csv(path);geo['geo_id']=ids;geo.iloc[[2,0,1]].to_csv(path,index=False)
    path=folder/'Tiny.rel';rel=pd.read_csv(path)
    for key in ('origin_id','destination_id'): rel[key]=rel[key].map(dict(enumerate(ids)))
    rel.to_csv(path,index=False)
    path=folder/'Tiny.dyna';frame=pd.read_csv(path)
    frame['other']=frame['entity_id']*100+frame['dyna_id']%48*.5
    frame['entity_id']=frame['entity_id'].map(dict(enumerate(ids)))
    if order=='time-major': frame=frame.sort_values(['time','entity_id'])
    elif order=='shuffled': frame=frame.sample(frac=1,random_state=19)
    frame.to_csv(path,index=False)
    path=folder/'config.json';raw=json.loads(path.read_text());raw['data_col']=['speed','other'];path.write_text(json.dumps(raw))
    return folder


def configuration(root,name='run'):
    return dict(model=dict(name='GRU',params=dict(hidden_size=4)),data=dict(dataset='Tiny',data_dir=str(root/'raw'),
        cache_dir=str(root/'cache'),cache=True,input_window=4,output_window=2,pad_training=False,scaler=dict(name='none')),
        training=dict(epochs=1,batch_size=8,shuffle=False),evaluation=dict(batch_size=4),
        runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def bundle(cfg): return prepare_traffic(cfg['data'],cfg['training'],cfg['evaluation'],17)


@pytest.mark.parametrize('text',[False,True])
@pytest.mark.parametrize('order',['blocks','time-major','shuffled'])
def test_actual_node_ids_and_sorted_clock_align_all_rows_features_graphs_and_warm_cache(tmp_path,text,order):
    folder=fixture(tmp_path,text,order);cfg=load_config(config=configuration(tmp_path));first=bundle(cfg)
    geo=pd.read_csv(folder/'Tiny.geo');frame=pd.read_csv(folder/'Tiny.dyna');times=sorted(frame['time'].unique())
    ids=geo['geo_id'].tolist();expected=np.empty((48,3,2))
    for time,stamp in enumerate(times):
        for node,value in enumerate(ids):
            row=frame[(frame['time']==stamp)&(frame['entity_id']==value)].iloc[0]
            expected[time,node]=row[['speed','other']].to_numpy(dtype=float)
    np.testing.assert_equal(first.train.dataset.history,expected)
    np.testing.assert_equal(first.train.dataset.targets,expected[...,:1])
    assert first.metadata.info['node_ids']==tuple(ids)
    assert first.metadata.input_layout['point_alignment']=='timestamp_entity_id_v2'
    assert np.all(np.diff(first.train.dataset.clock)==np.timedelta64(5,'m'))
    relation=pd.read_csv(folder/'Tiny.rel');matrix=np.zeros((3,3))
    for _,row in relation.iterrows(): matrix[ids.index(row['origin_id']),ids.index(row['destination_id'])]=1
    np.testing.assert_equal(first.metadata.adjacency,matrix)
    cached=bundle(cfg);assert cached.split_summary['cache']['status']=='hit'
    assert cached.metadata.info==first.metadata.info
    torch.testing.assert_close(cached.val.dataset[0]['X'],first.val.dataset[0]['X'],rtol=0,atol=0)


@pytest.mark.parametrize('problem',['duplicate','missing','unknown','missing-time','infinity','bad-feature','empty','duplicate-geo','missing-geo-id'])
def test_invalid_point_grids_and_node_identities_fail_before_training(tmp_path,problem):
    folder=fixture(tmp_path);path=folder/'Tiny.dyna';frame=pd.read_csv(path)
    if problem=='duplicate': frame=pd.concat((frame,frame.iloc[:1]))
    elif problem=='missing': frame=frame.iloc[1:]
    elif problem=='unknown': frame.loc[0,'entity_id']=999
    elif problem=='missing-time': frame.loc[0,'time']=None
    elif problem=='infinity': frame.loc[0,'speed']=float('inf')
    elif problem=='bad-feature': frame['speed']=frame['speed'].astype(object);frame.loc[0,'speed']='bad'
    elif problem=='empty': frame=frame.iloc[:0]
    else:
        geo=pd.read_csv(folder/'Tiny.geo')
        geo.loc[0,'geo_id']=geo.loc[1,'geo_id'] if problem=='duplicate-geo' else float('nan')
        geo.to_csv(folder/'Tiny.geo',index=False)
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError): bundle(load_config(config=configuration(tmp_path)))


def test_old_positional_cache_is_not_reused_and_remains_untouched(tmp_path):
    folder=fixture(tmp_path);cfg=load_config(config=configuration(tmp_path));first=bundle(cfg)
    with np.load(first.split_summary['cache']['path'],allow_pickle=False) as archive:
        fields={name:archive[name] for name in FIELDS}
    fields['history']=np.full_like(fields['history'],-999.)
    fields['target']=np.full_like(fields['target'],-999.)
    raw=json.loads((folder/'config.json').read_text())
    _,legacy=prepare_arrays(cfg['data'],raw,lambda:fields)
    path=Path(legacy['path']);before=path.stat().st_mtime_ns
    Path(first.split_summary['cache']['path']).unlink()
    current=bundle(cfg)
    assert current.split_summary['cache']['status']=='miss' and current.split_summary['cache']['path']!=legacy['path']
    assert current.train.dataset.history.max()>0 and path.stat().st_mtime_ns==before
    with np.load(path,allow_pickle=False) as archive: assert np.all(archive['history']==-999.)


def test_timezone_offsets_represent_one_chronological_utc_axis_and_nan_targets_survive(tmp_path):
    folder=fixture(tmp_path);path=folder/'Tiny.dyna';frame=pd.read_csv(path)
    frame['time']=pd.to_datetime(frame['time'],utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    frame.loc[0,'speed']=float('nan');frame.to_csv(path,index=False)
    data=bundle(load_config(config=configuration(tmp_path)))
    assert data.train.dataset.clock[0]==np.datetime64('2024-01-01T00:00:00','ns')
    assert np.isnan(data.train.dataset.targets).sum()==1


def test_standalone_weights_allow_row_shuffle_but_reject_changed_or_missing_node_alignment(tmp_path):
    folder=fixture(tmp_path);trained=run_experiment('train',config=configuration(tmp_path,'train'))
    path=folder/'Tiny.dyna';pd.read_csv(path).sample(frac=1,random_state=31).to_csv(path,index=False)
    evaluated=run_experiment('evaluate',config=configuration(tmp_path,'shuffled'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    old=deepcopy(torch.load(trained.artifacts['best'],weights_only=True))
    def strip(value):
        if isinstance(value,dict):
            return {key:strip(item) for key,item in value.items() if key not in ('point_alignment','node_ids')}
        if isinstance(value,list): return [strip(item) for item in value]
        return value
    old._metadata=strip(old._metadata)
    legacy=tmp_path/'old-without-alignment.pt';torch.save(old,legacy)
    with pytest.raises(ValueError,match='input|Input'):
        run_experiment('evaluate',config=configuration(tmp_path,'old'),checkpoint=legacy)
    path=folder/'Tiny.geo';geo=pd.read_csv(path);geo.iloc[::-1].to_csv(path,index=False)
    with pytest.raises(ValueError,match='input|Input'):
        run_experiment('evaluate',config=configuration(tmp_path,'wrong-nodes'),checkpoint=trained.artifacts['best'])
