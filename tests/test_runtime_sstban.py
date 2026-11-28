"""Correct head widths, explicit mask semantics, calendar data and SSTBAN lifecycle."""
from copy import deepcopy
import importlib
from types import MethodType,SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from libcity.config.runtime import load_config,merge_config
from libcity.data.bundle import DatasetMetadata
from libcity.executor.sstban_step import SSTBANLoss,auxiliary_step,block_keep_mask,consistency_cost,train
from libcity.model.traffic_flow_prediction.runtime_sstban import SSTBAN,MAB
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,sstban_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark=pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')
SMALL=dict(L=1,K=2,d=3,reference=2,bn_decay=.1)


def metadata(nodes=4,features=2,outputs=2,slots=24):
    calendar=dict(kind='calendar_positions',columns=['weekday','time_slot'],steps_per_day=slots,time_basis='UTC',weekday_origin='Monday')
    return DatasetMetadata('tiny',nodes,features,outputs,info=dict(time_intervals=86400//slots),
        input_layout=dict(kind='window',history_features=calendar,future_features=calendar))


def inputs(meta,batch,history,horizon):
    times=torch.arange(history+horizon)[None].expand(batch,-1)
    tokens=torch.stack((times%7,times%meta.input_layout['history_features']['steps_per_day']),-1)
    return dict(X=torch.randn(batch,history,meta.num_nodes,meta.feature_dim,dtype=torch.float64,requires_grad=True),
        history_features=tokens[:,:history],future_features=tokens[:,history:])


def independent_attention(query,key,value,allowed=None):
    score=query@key.transpose(-1,-2)/np.sqrt(query.shape[-1])
    if allowed is None: return score.softmax(-1)@value
    allowed=allowed.expand_as(score)
    masked=score.masked_fill(~allowed,-torch.inf)
    masked=torch.where(allowed.any(-1,keepdim=True),masked,torch.zeros_like(masked))
    return (masked.softmax(-1)*allowed)@value


def reference_mab(self,query,key,batch_size,type='spatial',mask=None):
    query,key,value=self.FC_q(query),self.FC_k(key),self.FC_v(key)
    results=[]
    for head in range(self.K):
        sl=slice(head*self.d,(head+1)*self.d)
        q,k,v=query[...,sl],key[...,sl],value[...,sl]
        kept=None if mask is None else mask[...,sl].bool().any(-1)
        if type=='temporal':
            q,k,v=[item.transpose(1,2) for item in (q,k,v)]
            if kept is not None: kept=kept.transpose(1,2)
        allowed=None if kept is None else kept.unsqueeze(-2 if self.mask_role=='key' else -1)
        result=torch.stack([independent_attention(q[i],k[i],v[i],None if allowed is None else allowed[i]) for i in range(batch_size)])
        results.append(result.transpose(1,2) if type=='temporal' else result)
    return self.FC(torch.cat(results,-1))


def reference_transform(self,source,history,future):
    q,k,v=self.FC_q(future),self.FC_k(history),self.FC_v(source);result=[]
    for head in range(self.K):
        sl=slice(head*self.d,(head+1)*self.d)
        result.append(independent_attention(q[...,sl].transpose(1,2),k[...,sl].transpose(1,2),v[...,sl].transpose(1,2)).transpose(1,2))
    return self.FC(torch.cat(result,-1))


def corrected_reference(old):
    def embedding(self,spatial,tokens,slots):
        tokens=tokens.long()
        values=torch.cat((F.one_hot(tokens[...,0],7),F.one_hot(tokens[...,1],slots)),-1).to(spatial.dtype)
        return spatial[None,None]+self.FC_te(values[:,:,None])
    old.STEmbedding.forward=MethodType(embedding,old.STEmbedding)
    for block in [*old.STAttBlock_1,*old.STAttBlock_2,*old.STAttBlock_self]:
        for attention in (block.spatialAttention,block.temporalAttention):
            for name,role in (('mab0','key'),('mab1','query')):
                module=getattr(attention,name);module.mask_role=role;module.forward=MethodType(reference_mab,module)
    old.transformAttention.forward=MethodType(reference_transform,old.transformAttention)


@pytest.mark.parametrize('case',[(2,4,3,2,5,3,{}),(2,3,1,1,12,2,dict(L=3,K=16,d=8,reference=3)),
    (2,3,2,2,3,4,dict(K=2,d=2,reference=3)),(1,1,1,1,1,1,dict(K=1,d=1,reference=1))],
    ids=['multioutput-unequal-head-width','defaults','equal-heads-ambiguous-reference-size','singletons'])
def test_corrected_original_full_prediction_aux_features_all_gradients_and_bn(case,monkeypatch):
    batch,nodes,features,outputs,history,horizon,changes=case
    torch.set_num_threads(1);torch.manual_seed(17);params={**SMALL,**changes}
    meta=metadata(nodes,features,outputs);model=SSTBAN(meta,history,horizon,**params).double()
    legacy=importlib.import_module('libcity.model.traffic_flow_prediction.SSTBAN')
    old=legacy.SSTBAN(dict(params,input_window=history,output_window=horizon,T_miss_len=1,node_miss_rate=.5,
        in_channels=features,out_channels=outputs),dict(num_nodes=nodes,output_dim=outputs,time_intervals=3600)).double()
    old.load_state_dict(model.state_dict());corrected_reference(old)
    if nodes==1: old.eval();model.eval()
    batch_inputs=inputs(meta,batch,history,horizon);other={key:value.detach().clone().requires_grad_(key=='X') for key,value in batch_inputs.items()}
    mask=torch.rand(batch,history,nodes,params['K']*params['d'])>.5
    if nodes>1: mask[0]=False
    monkeypatch.setattr(legacy.np.random,'rand',lambda *shape:(~mask).numpy().astype(float))
    expected=old(other['X'],torch.cat((other['history_features'],other['future_features']),1),'train')
    actual=model.forward_with_aux(batch_inputs,mask)
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=3e-7,atol=2e-8)
    sum(value.square().mean()*(index+1) for index,value in enumerate(actual)).backward()
    sum(value.square().mean()*(index+1) for index,value in enumerate(expected)).backward()
    torch.testing.assert_close(batch_inputs['X'].grad,other['X'].grad,rtol=2e-5,atol=2e-7)
    for name,value in model.named_parameters():
        assert value.grad is not None,name
        torch.testing.assert_close(value.grad,dict(old.named_parameters())[name].grad,rtol=3e-5,atol=2e-7)
    for name,value in model.named_buffers(): torch.testing.assert_close(value,old.state_dict()[name],rtol=1e-9,atol=1e-9)


@pytest.mark.parametrize('temporal',[False,True])
@pytest.mark.parametrize('role',['key','query'])
def test_explicit_mask_roles_correct_heads_and_fully_missing_rows_match_independent_attention(temporal,role):
    torch.manual_seed(17);module=MAB(2,3,6,6,.1,temporal,role).double()
    module.FC_q=module.FC_k=module.FC_v=module.FC=nn.Identity()
    shape_q=(2,2,3,6) if temporal else (2,3,2,6)
    shape_k=(2,4,3,6) if temporal else (2,3,4,6)
    query=torch.randn(shape_q,dtype=torch.float64,requires_grad=True);key=torch.randn(shape_k,dtype=torch.float64,requires_grad=True)
    mask=torch.rand(shape_k if role=='key' else shape_q)>.3;mask[0]=False
    reference=SimpleNamespace(K=2,d=3,mask_role=role,FC_q=nn.Identity(),FC_k=nn.Identity(),FC_v=nn.Identity(),FC=nn.Identity())
    expected=reference_mab(reference,query,key,2,'temporal' if temporal else 'spatial',mask)
    actual=module(query,key,mask)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert torch.count_nonzero(actual[0])==0
    first=torch.autograd.grad(actual.square().sum(),(query,key),retain_graph=True)
    second=torch.autograd.grad(expected.square().sum(),(query,key))
    for a,b in zip(first,second): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)


@pytest.mark.parametrize('probability',[0.,.4,1.])
def test_block_masks_cover_short_and_nondivisible_histories_and_have_explicit_keep_meaning(probability):
    torch.manual_seed(17);mask=block_keep_mask((20,5,10,12),3,probability,'cpu')
    assert mask.shape==(20,5,10,12) and mask.dtype==torch.bool
    assert torch.equal(mask[:,0],mask[:,1]) and torch.equal(mask[:,0],mask[:,2]) and torch.equal(mask[:,3],mask[:,4])
    if probability in (0.,1.): assert bool(mask.all())==(probability==1.) and bool(mask.any())==(probability==1.)
    else: assert abs(float(mask.float().mean())-probability)<.03
    assert block_keep_mask((1,1,1,2),12,probability,'cpu').shape==(1,1,1,2)


def configuration(root,name='run'):
    return dict(model=dict(name='SSTBAN',params=SMALL),
        data=dict(dataset='BottleneckTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),input_window=5,output_window=3,pad_training=False),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,loss=dict(self_weight=.4),strategy=dict(block_length=3,keep_probability=.6),
            scheduler=dict(name='plateau',interval='metric',monitor='val_loss',factor=.5,patience=0)),
        evaluation=dict(batch_size=4),runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def equal(left,right):
    if isinstance(left,torch.Tensor): torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True): equal(a,b)
    else: assert left==right


@pytest.mark.parametrize('scaler',['standard','minmax01'])
def test_actual_mask_bn_training_pure_mae_evaluation_and_exact_resume(tmp_path,scaler):
    sstban_fixture(tmp_path/'raw')
    def config(name):
        cfg=configuration(tmp_path,name);cfg['data']['scaler']=dict(name=scaler);return cfg
    full=run_experiment('train',config=config('full'))
    evaluated=run_experiment('evaluate',config=config('evaluated'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as result:
        prediction,truth=result['prediction'],result['truth'];valid=np.isfinite(truth)&(truth!=0)
        assert prediction.shape==(10,3,4,2)
        assert evaluated.metrics['loss']==pytest.approx(np.abs(prediction[valid]-truth[valid]).mean(),rel=2e-6)
    assert not (tmp_path/'runs/evaluated/checkpoints').exists() and not (tmp_path/'runs/evaluated/metrics.jsonl').exists()
    partial=make_context(config('partial'))
    try: train(partial,until_epoch=1)
    finally: partial.tracker.close()
    resumed=run_experiment('train',config=config('resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual=[torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'): equal(expected[key],actual[key])


@pytest.mark.parametrize('low_precision',[False,True])
def test_consistency_cost_valid_target_weighting_microgradients_and_exact_counts(low_precision):
    torch.manual_seed(17);nodes=130
    loss=SSTBANLoss(SimpleNamespace(metadata=SimpleNamespace(output_dim=1),scaler=NoneScaler()),dict(name='masked_mae',mask_value=0.,self_weight=.4))
    class Probe(nn.Module):
        def __init__(self):
            super().__init__();self.steps,self.nodes,self.hidden_dim=1,nodes,2
            self.prediction=nn.Parameter(torch.randn(3,2,nodes,1,dtype=torch.float64))
            self.complete=nn.Parameter(torch.randn(3,1,nodes,2,dtype=torch.float64))
            self.corrupted=nn.Parameter(torch.randn(3,1,nodes,2,dtype=torch.float64))
            with torch.no_grad(): self.corrupted[0,0,0]=0
        def forward_with_aux(self,batch,mask):
            index=batch['X'][:,0,0,0].long();dtype=torch.bfloat16 if low_precision else torch.float64
            return tuple(value[index].to(dtype) for value in (self.prediction,self.complete,self.corrupted))
    model=Probe();step=auxiliary_step(loss,dict(block_length=3,keep_probability=.6))
    raw=torch.ones(3,2,nodes,1,dtype=torch.float64);raw[0].flatten()[:3]=0;raw[1].flatten()[:1]=0;raw[2]=float('nan')
    batch=dict(X=torch.arange(3,dtype=torch.float64)[:,None,None,None].expand(-1,1,nodes,1),y=raw,y_raw=raw)
    actual=step(model,batch,None);assert actual.weight==516
    prediction,complete,corrupted=model.forward_with_aux(batch,None);costs=[]
    for i in range(3):
        dtype=torch.float32 if low_precision else torch.float64
        valid=corrupted[i]!=0
        costs.append((complete[i].to(dtype)[valid]-corrupted[i].to(dtype)[valid]).square().mean())
    costs=torch.stack(costs);main=loss(prediction,raw,raw)
    expected=.6*main.loss+.4*(costs*costs.new_tensor([257,259,0])).sum()/516
    torch.testing.assert_close(actual.loss,expected,rtol=1e-7 if low_precision else 1e-12,atol=1e-8 if low_precision else 1e-12)
    if not low_precision:
        micros=[step(model,{key:value[i:i+1] for key,value in batch.items()},None) for i in range(3)]
        combined=sum(value.loss*value.weight for value in micros)/516
        first=torch.autograd.grad(actual.loss,tuple(model.parameters()),retain_graph=True)
        second=torch.autograd.grad(combined,tuple(model.parameters()))
        for a,b in zip(first,second): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
    assert torch.count_nonzero(consistency_cost(torch.zeros(2,1,2,2),torch.zeros(2,1,2,2)))==0


def test_calendar_raw_missing_cache_future_isolation_and_pure_inference(tmp_path,monkeypatch):
    folder=sstban_fixture(tmp_path/'raw');path=folder/'BottleneckTiny.dyna';frame=pd.read_csv(path);times=sorted(frame['time'].unique())
    frame.loc[(frame['entity_id']==71)&frame['time'].isin([times[35],times[36]]),'value_0']=float('nan')
    frame.loc[(frame['entity_id']==3)&(frame['time']==times[36]),'value_0']=0.;frame.to_csv(path,index=False)
    context=make_context(configuration(tmp_path,'data'))
    try:
        bundle=context.bundle;sample=bundle.val.dataset[0];scaler=deepcopy(vars(bundle.scaler))
        assert sample['history_features'].dtype==sample['future_features'].dtype==torch.int64
        np.testing.assert_array_equal(sample['history_features'],np.array([[1,i] for i in range(7,12)]))
        np.testing.assert_array_equal(sample['future_features'],np.array([[1,i] for i in range(12,15)]))
        assert torch.isfinite(sample['X']).all() and torch.isnan(sample['y_raw']).any() and torch.any(sample['y_raw']==0)
        batch=next(iter(bundle.val));model=context.model.eval();before=torch.get_rng_state().clone()
        prediction=model(batch);torch.testing.assert_close(prediction,model.forward_with_aux(batch,torch.ones(4,5,4,6,dtype=torch.bool))[0],rtol=0,atol=0)
        assert torch.equal(before,torch.get_rng_state())
        assert not any(hasattr(model,name) for name in ('device','logger','scaler','_scaler','X_miss','complete_X_enc','mode'))
        def forbidden(*args,**kwargs): raise AssertionError('auxiliary branch executed')
        monkeypatch.setattr(model.STAttBlock_self[0],'forward',forbidden)
        assert model({**batch,'y':torch.full_like(batch['y'],float('nan'))}).shape==(4,3,4,2)
        with pytest.raises(AssertionError,match='auxiliary'): model.forward_with_aux(batch,torch.ones(4,5,4,6,dtype=torch.bool))
        separate=torch.cat([model({key:value[i:i+1] for key,value in batch.items()}) for i in range(4)])
        torch.testing.assert_close(prediction,separate,rtol=2e-6,atol=1e-6)
        frame.loc[frame['time']>=times[36],['value_0','value_1']]+=1000;frame.to_csv(path,index=False)
        changed=make_context(configuration(tmp_path,'changed'))
        try:
            assert vars(changed.bundle.scaler)==scaler
            for key in ('X','history_features','future_features'): torch.testing.assert_close(sample[key],changed.bundle.val.dataset[0][key],rtol=0,atol=0)
        finally: changed.tracker.close()
        warm=make_context(configuration(tmp_path,'warm'))
        try: assert warm.bundle.split_summary['cache']['status']=='hit'
        finally: warm.tracker.close()
    finally: context.tracker.close()


def test_zero_auxiliary_weight_skips_mask_sampling_and_auxiliary_forward(tmp_path,monkeypatch):
    sstban_fixture(tmp_path/'raw');cfg=configuration(tmp_path,'plain');cfg['training']['loss']['self_weight']=0.
    context=make_context(cfg)
    try:
        def forbidden(*args,**kwargs): raise AssertionError('auxiliary work executed')
        module=importlib.import_module('libcity.executor.sstban_step');monkeypatch.setattr(module,'block_keep_mask',forbidden)
        monkeypatch.setattr(context.model,'forward_with_aux',forbidden)
        result=auxiliary_step(context.loss_fn,context.config['training']['strategy'])(context.model,next(iter(context.bundle.train)),None)
        assert result.weight>0 and torch.isfinite(result.loss)
    finally: context.tracker.close()


@pytest.mark.parametrize('singleton',[False,True])
def test_cpu_amp_real_singleton_tail_and_empty_attention(tmp_path,singleton):
    if singleton: atomic_fixture(tmp_path/'raw',name='BottleneckTiny',steps=60,nodes=1)
    else: sstban_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,'amp');cfg['training'].update(epochs=1,amp=True,batch_size=11)
    cfg['training']['strategy']['keep_probability']=0.
    if singleton: cfg['data'].update(input_window=1,output_window=2)
    result=run_experiment('train',config=cfg);assert np.isfinite(list(result.metrics.values())).all()
    state=torch.load(result.artifacts['last'],weights_only=True)
    if singleton: assert state['model']['FC_1.convs.0.batch_norm.num_batches_tracked']==3


def test_windows_spawn_calendar_tokens(tmp_path):
    sstban_fixture(tmp_path/'raw');cfg=configuration(tmp_path,'spawn');cfg['training']['num_workers']=1
    context=make_context(cfg)
    try:
        batch=next(iter(context.bundle.train));assert batch['history_features'].shape==(5,5,2) and batch['future_features'].shape==(5,3,2)
    finally: context.tracker.close()


@pytest.mark.parametrize('change',[
    dict(model=dict(params=dict(L=0))),dict(model=dict(params=dict(K=0))),dict(model=dict(params=dict(d=0))),
    dict(model=dict(params=dict(reference=0))),dict(model=dict(params=dict(bn_decay=0.))),dict(model=dict(params=dict(bn_decay=1.1))),
    dict(model=dict(params=dict(in_channels=1))),dict(data=dict(graph=dict(name='binary'))),dict(data=dict(graph=dict(symmetric=True))),
    dict(data=dict(features=dict(time_in_day=True))),dict(data=dict(features=dict(known_future_columns=['weather']))),
    dict(training=dict(strategy=dict(block_length=0))),dict(training=dict(strategy=dict(keep_probability=1.1))),
    dict(training=dict(loss=dict(self_weight=-.1))),dict(training=dict(loss=dict(self_weight=True)))])
def test_invalid_sstban_configuration_rejected_before_data(change):
    with pytest.raises((ValueError,TypeError)):
        load_config(config=merge_config(dict(model=dict(name='SSTBAN'),data=dict(dataset='not-read')),change))
