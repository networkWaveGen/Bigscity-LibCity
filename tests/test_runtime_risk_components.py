"""GSNet prerequisites: partial graphs, risk weights and batch-stable rankings."""
from copy import deepcopy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.data.gsnet_bundle import prepare_gsnet
from libcity.data.sources import atomic_sources
from libcity.evaluator.runtime_risk import RiskMetrics
from libcity.executor.risk_loss import RiskLoss
from libcity.executor.risk_evaluation import evaluate_risk
from libcity.executor.train import build_context,train_loop
from libcity.pipeline.experiment import initialize_runtime
from libcity.testing.fixtures import risk_fixture
from libcity.utils.normalization import NoneScaler,StandardScaler,MinMax01Scaler

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,name='probe'):
    cfg = load_config(config=dict(model=dict(name='GRU',params=dict(hidden_size=4)),
        data=dict(dataset='RiskTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),input_window=3,output_window=2,
            pad_training=False,scaler=dict(name='minmax01'),features=dict(load_external=True,time_in_day=True,day_in_week=True,known_future_columns=['holiday'])),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2),evaluation=dict(batch_size=4),
        runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs'))))
    # Component API tests; the GSNet neural model is not registered by this batch.
    cfg['data']['sampling']=dict(name='risk_cpt',closeness=2,period=1,trend=0,interval_period=1,interval_trend=1)
    cfg['data']['graph']=dict(name='risk_multigraph',use_poi=True)
    cfg['data']['features'].update(graph_columns=None,external_scaler=dict(name='minmax01'))
    cfg['training']['loss']=dict(name='risk_mse',mask_value=None,thresholds=None,weights=None)
    cfg['evaluation'].update(metrics=['MAE','RMSE','Precision','Recall','F1-Score','MAP','PCC'],mask_value=None,topk=2)
    return cfg


def bundle(cfg):
    return prepare_gsnet(cfg['data'],cfg['training'],cfg['evaluation'],cfg['runtime']['seed'])


@pytest.mark.parametrize('poi',[False,True])
def test_partial_noncontiguous_graph_ids_map_to_full_grid_and_preserve_graph_views(tmp_path,poi):
    risk_fixture(tmp_path/'raw',poi=poi)
    cfg=configuration(tmp_path);result=bundle(cfg)
    assert result.metadata.num_nodes==6 and result.metadata.input_layout['graph_nodes']==3
    assert result.metadata.info['graph_node_ids']==(71,3,108)
    static=result.metadata.static_data
    np.testing.assert_equal(static['risk_mask'],[1.,0.,2.,0.,0.,.5])
    expected=np.zeros((6,3));expected[[5,0,2],[0,1,2]]=1
    np.testing.assert_equal(static['grid_node_map'],expected)
    np.testing.assert_equal(static['road_adj'],[[1,1,0],[1,1,1],[0,1,1]])
    np.testing.assert_allclose(static['risk_adj'],.3*np.arange(1,4)[:,None]/np.arange(1,4)[None],rtol=1e-7)
    assert ('poi_adj' in static)==poi and not static['grid_node_map'].flags.writeable
    assert result.metadata.input_layout['graph_input_indices']==(0,1)
    sample=result.val.dataset[0];start=int(result.val.dataset.target_starts[0])
    assert sample['X'].shape==(3,6,3) and sample['future_features'].shape==(2,32)
    assert sample['future_features'][0,start%24]==1
    assert sample['history_features'][0,(start-1)%24]==1
    assert sample['future_features'][0,:24].sum()==1
    assert any(row['name']=='RiskTiny.grid' for row in atomic_sources(cfg['data']))
    assert not any(row['name']=='RiskTiny.dyna' and row['size'] is None for row in atomic_sources(cfg['data']))


def test_known_future_time_flags_and_train_only_numeric_scalers_are_separate_from_risk_targets(tmp_path):
    folder=risk_fixture(tmp_path/'raw');cfg=configuration(tmp_path)
    cfg['data']['features']['known_future_columns']=['holiday','value_2']
    first=bundle(cfg);sample=first.val.dataset[0]
    assert first.metadata.input_layout['future_feature_names'][-1]=='value_2'
    assert sample['future_features'].shape==(2,33)
    raw=first.train.dataset.arrays['history'];assert first.scaler.min==pytest.approx(np.nanmin(raw[:57,:,:1]))
    expected=pd.read_csv(folder/'RiskTiny.ext').sort_values('time').iloc[:57]['temperature'].max()
    assert first.metadata.input_layout['external_scalers']['temperature']['state']['max']==pytest.approx(expected)
    frame=pd.read_csv(folder/'RiskTiny.grid');times=sorted(frame['time'].unique())
    frame.loc[frame['time'].isin(times[57:]),'value_0']+=1000;frame.to_csv(folder/'RiskTiny.grid',index=False)
    external=pd.read_csv(folder/'RiskTiny.ext');external.loc[external['time'].isin(times[57:]),'temperature']+=500
    external.to_csv(folder/'RiskTiny.ext',index=False)
    changed=bundle(cfg)
    assert first.metadata.input_layout['external_scalers']==changed.metadata.input_layout['external_scalers']
    assert vars(first.scaler)==vars(changed.scaler)
    for name in ('X','history_features','future_features'): torch.testing.assert_close(sample[name],changed.val.dataset[0][name],rtol=0,atol=0)
    assert not torch.equal(sample['y_raw'],changed.val.dataset[0]['y_raw'])
    sets=[{int(start+step) for start in loader.dataset.target_starts for step in range(2)} for loader in (first.train,first.val,first.test)]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]


@pytest.mark.parametrize('problem',['target','spatial-observation','missing-dimensions','unknown-id','duplicate-edge','empty-mask'])
def test_invalid_risk_data_or_future_declarations_fail_clearly(tmp_path,problem):
    folder=risk_fixture(tmp_path/'raw');cfg=configuration(tmp_path)
    if problem=='target': cfg['data']['features']['known_future_columns']=['value_0']
    elif problem=='spatial-observation': cfg['data']['features']['known_future_columns']=['value_1']
    elif problem=='missing-dimensions':
        path=folder/'config.json';raw=json.loads(path.read_text());del raw['grid_len_row'];path.write_text(json.dumps(raw))
    elif problem in ('unknown-id','duplicate-edge'):
        path=folder/'RiskTiny.rel';frame=pd.read_csv(path)
        if problem=='unknown-id': frame.loc[0,'origin_id']=9999
        else: frame=pd.concat((frame,frame.iloc[:1]))
        frame.to_csv(path,index=False)
    else:
        path=folder/'RiskTiny.geo';frame=pd.read_csv(path);frame['risk_mask']=0.;frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='GSNet|Risk|risk'): bundle(cfg)


def risk_loss(scaler,thresholds=None,weights=None):
    meta=DatasetMetadata('tiny',6,1,1,static_data=dict(risk_mask=np.array([1.,0.,2.,0.,0.,.5])),
        info=dict(risk_thresholds=[0.,20.,28.],risk_weights=[.05,.2,.25,.5]))
    return RiskLoss(SimpleNamespace(metadata=meta,scaler=scaler),dict(name='risk_mse',mask_value=None,
        thresholds=thresholds,weights=weights))


@pytest.mark.parametrize('scaler',[NoneScaler(),StandardScaler(10.,2.),MinMax01Scaler(5.,35.)],ids=['none','standard','minmax'])
@pytest.mark.parametrize('missing',[False,True])
def test_raw_spatial_square_weights_and_threshold_ties_match_independent_objective_and_gradients(scaler,missing):
    torch.manual_seed(17)
    target=torch.tensor([0.,20.,28.,3.,5.,29.],dtype=torch.float64).reshape(1,1,6,1).expand(2,3,-1,-1).clone()
    if missing: target[0,1,2,0]=float('nan')
    prediction=torch.randn_like(target,requires_grad=True)
    loss=risk_loss(scaler);actual=loss(prediction,scaler.transform(target),target)
    raw=scaler.inverse_transform(prediction)
    region=np.array([1.,0.,2.,0.,0.,.5]);terms=[]
    for batch in range(2):
        for horizon in range(3):
            for cell in (0,2,5):
                truth=target[batch,horizon,cell,0]
                if not torch.isfinite(truth): continue
                weight=.05 if truth<=0 else .2 if truth<=20 else .25 if truth<=28 else .5
                terms.append((raw[batch,horizon,cell,0]-truth).square()*(region[cell]/region.mean())**2*weight)
    expected=torch.stack(terms).mean()*.5
    assert actual.weight==len(terms)
    torch.testing.assert_close(actual.loss,expected,rtol=1e-13,atol=1e-10)
    first=torch.autograd.grad(actual.loss,prediction,retain_graph=True)[0]
    second=torch.autograd.grad(expected,prediction)[0]
    torch.testing.assert_close(first,second,rtol=1e-13,atol=1e-11)


def test_empty_thresholds_constant_weights_and_no_known_active_targets_have_finite_zero_gradients():
    loss=risk_loss(NoneScaler(),[],[1.])
    prediction=torch.ones(2,3,6,1,dtype=torch.float64,requires_grad=True)
    target=torch.full_like(prediction,float('nan'));target[:,:,1]=0.
    result=loss(prediction,target,target)
    assert result.weight==0 and result.loss.item()==0
    result.loss.backward();assert torch.count_nonzero(prediction.grad)==0
    result=loss(prediction,torch.zeros_like(prediction),torch.zeros_like(prediction))
    assert result.weight==18 and result.loss>0


def test_weighted_risk_gradients_accumulate_by_known_region_count_across_unequal_batches():
    torch.manual_seed(17)
    prediction=torch.randn(3,2,6,1,dtype=torch.float64,requires_grad=True)
    target=torch.rand_like(prediction)*40;target[0,0,2]=float('nan')
    loss=risk_loss(StandardScaler(10.,2.));full=loss(prediction,target,target)
    expected=torch.autograd.grad(full.loss,prediction)[0]
    total=0
    for part in (slice(0,1),slice(1,None)):
        result=loss(prediction[part],target[part],target[part]);(result.loss*result.weight).backward();total+=result.weight
    torch.testing.assert_close(prediction.grad/total,expected,rtol=1e-13,atol=1e-12)


def test_raw_risk_cache_reuses_graph_selection_and_external_scaler_without_rereading_observations(tmp_path,monkeypatch):
    risk_fixture(tmp_path/'raw');cfg=configuration(tmp_path)
    first=bundle(cfg);assert first.split_summary['cache']['status']=='miss'
    cfg['data']['features'].update(graph_columns=['value_1'],external_scaler=dict(name='none'))
    cfg['data']['graph']['use_poi']=False
    cfg['data'].update(scaler=dict(name='standard'),output_window=1)
    original=pd.read_csv
    def guarded(path,*args,**kwargs):
        if Path(path).suffix in ('.grid','.ext') and kwargs.get('nrows')!=0: pytest.fail('Raw cache hit re-read time-series rows')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',guarded)
    changed=bundle(cfg)
    assert changed.split_summary['cache']['status']=='hit'
    assert changed.split_summary['cache']['path']==first.split_summary['cache']['path']
    assert changed.metadata.input_layout['graph_input_indices']==(1,)
    assert 'poi_adj' not in changed.metadata.static_data


@pytest.mark.parametrize('thresholds,weights',[([1.,1.],[1.,2.,3.]),([1.],[1.]),([1.],[0.,0.]),([1.],[-1.,2.])])
def test_invalid_risk_bin_configuration_is_rejected(thresholds,weights):
    with pytest.raises(ValueError,match='Risk'): risk_loss(NoneScaler(),thresholds,weights)


@pytest.mark.parametrize('mode',['single','average'])
@pytest.mark.parametrize('topk',[1,10])
def test_rankings_and_spatial_regression_match_numpy_across_batches_and_ties(mode,topk):
    rng=np.random.default_rng(17)
    target=rng.integers(0,4,size=(5,3,6,1)).astype(float)
    forecast=rng.integers(0,3,size=target.shape).astype(float)
    target[0,0,2,0]=np.nan;target[1,1]=0;forecast[2,2]=1
    region=np.array([1.,0.,2.,0.,0.,.5])
    names=['MAE','RMSE','Precision','Recall','F1-Score','MAP','PCC']
    accumulator=RiskMetrics(names,region,topk,mode=mode)
    for selection in (slice(0,2),slice(2,3),slice(3,5)):
        accumulator.update(torch.tensor(forecast[selection]),torch.tensor(target[selection]))
    actual=accumulator.compute()
    for horizon in range(3):
        frames=range(horizon+1) if mode=='average' else [horizon]
        scores=[];errors=[]
        for batch in range(5):
            for time in frames:
                valid=np.isfinite(target[batch,time,:,0])&(region>0)
                ids=np.flatnonzero(valid);truth=target[batch,time,ids,0];pred=forecast[batch,time,ids,0]
                errors.extend(pred-truth)
                order=np.lexsort((ids,-pred))[:topk];hit=(truth[order]>0).astype(float)
                precision=hit.sum()/len(order) if len(order) else 0.;positives=(truth>0).sum()
                recall=hit.sum()/positives if positives else 0.
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.
                ap=(np.cumsum(hit)/np.arange(1,len(hit)+1)*hit).sum()/min(positives,len(order)) if positives and len(order) else 0.
                pcc=np.corrcoef(truth,pred)[0,1] if len(truth)>1 and truth.std()>0 and pred.std()>0 else 0.
                scores.append([precision,recall,f1,ap,pcc])
        expected=np.mean(scores,0)
        for name,value in zip(names[2:],expected): assert actual[f'{name}@{horizon+1}']==pytest.approx(value,abs=2e-12)
        assert actual[f'MAE@{horizon+1}']==pytest.approx(np.abs(errors).mean(),abs=2e-12)
        assert actual[f'RMSE@{horizon+1}']==pytest.approx(np.sqrt(np.square(errors).mean()),abs=2e-12)


class RiskProbe(torch.nn.Module):
    def __init__(self,features,future,horizon):
        super().__init__();self.projection=torch.nn.Linear(features,1);self.clock=torch.nn.Linear(future,1)
        self.dropout=torch.nn.Dropout(.2);self.horizon=horizon

    def forward(self,inputs):
        assert 'y' not in inputs and 'y_raw' not in inputs
        values=self.projection(self.dropout(inputs['X'][:,0]))+self.clock(inputs['future_features'][:,0])[:,None]
        return values[:,None].expand(-1,self.horizon,-1,-1)


def equal(left,right):
    if isinstance(left,torch.Tensor): torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True): equal(a,b)
    else: assert left==right


def probe_context(root,name,amp=False):
    cfg=configuration(root,name);cfg['training']['amp']=amp;initialize_runtime(cfg['runtime']);data=bundle(cfg)
    args=dict(features=3,future=data.metadata.input_layout['future_external_dim'],horizon=2)
    cfg['model']=dict(name='RiskProbe',params=args)
    directory=root/'runs'/name;directory.mkdir(parents=True)
    return build_context(RiskProbe(**args),data,cfg,torch.device('cpu'),directory,logging.getLogger('risk-probe'),
        loss_fn=RiskLoss(data,cfg['training']['loss']),evaluate_fn=evaluate_risk)


def test_actual_risk_components_train_export_and_full_resume_without_claiming_native_gsnet(tmp_path):
    folder=risk_fixture(tmp_path/'raw')
    context=lambda name:probe_context(tmp_path,name)
    contexts=[]
    try:
        full=context('full');contexts.append(full);train_loop(full)
        result=evaluate_risk(full.model,full.bundle.test,device=full.device,loss_fn=full.loss_fn,evaluation=full.config['evaluation'],directory=full.directory/'eval')
        with np.load(result.artifacts['predictions'],allow_pickle=False) as archive:
            assert archive['prediction'].shape[1:]==(2,6,1)
            np.testing.assert_equal(archive['regions'],[True,False,True,False,False,True])
            np.testing.assert_equal(archive['truth'][0],full.bundle.test.dataset[0]['y_raw'])
        evaluation=dict(full.config['evaluation'],save_predictions=False)
        no_export=evaluate_risk(full.model,full.bundle.test,device=full.device,loss_fn=full.loss_fn,evaluation=evaluation,directory=full.directory/'eval')
        assert no_export.metrics==result.metrics and 'predictions' not in no_export.artifacts
        assert not (full.directory/'eval/predictions.npz').exists()
        partial=context('partial');contexts.append(partial);train_loop(partial,until_epoch=1)
        resumed=context('resumed');contexts.append(resumed)
        resumed.checkpoints.load_last(partial.directory/'checkpoints/last.pt',resumed);train_loop(resumed)
        expected,actual=[torch.load(item.directory/'checkpoints/last.pt',weights_only=True) for item in (full,resumed)]
        for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'): equal(expected[key],actual[key])
        relation=pd.read_csv(folder/'RiskTiny.rel');relation['risk']+=.01;relation.to_csv(folder/'RiskTiny.rel',index=False)
        changed=context('changed');contexts.append(changed)
        with pytest.raises(ValueError,match='source files changed'): changed.checkpoints.load_last(partial.directory/'checkpoints/last.pt',changed)
    finally:
        for item in contexts: item.tracker.close()


def test_actual_risk_weighted_loss_and_rankings_support_cpu_amp_and_real_tail(tmp_path):
    risk_fixture(tmp_path/'raw');context=probe_context(tmp_path,'amp',amp=True)
    try:
        score=train_loop(context,until_epoch=1)
        assert np.isfinite(score) and context.state.optimizer_steps>0
    finally: context.tracker.close()
