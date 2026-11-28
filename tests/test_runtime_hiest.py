"""HIEST's overlapping regions, sample-isolated graphs and auxiliary training."""
from copy import deepcopy
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config,merge_config
from libcity.data.bundle import DatasetMetadata
from libcity.data.hiest_bundle import prepare_hiest,region_mapping
from libcity.executor.hiest_step import HIESTLoss,auxiliary_costs,auxiliary_step
from libcity.executor.train import train_loop
from libcity.model.traffic_speed_prediction.runtime_hiest import HIEST,global_supports
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import hiest_fixture,atomic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark=pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


SMALL=dict(blocks=1,layers=2,apt_layer=False,kernel_size=2,residual_channels=4,dilation_channels=4,
    skip_channels=6,end_channels=8,global_nodes=3,adjtype='doubletransition',order=2,dropout=0.,n1=1.,n2=1.,n3=1.,n4=1.)


def metadata(nodes=5,features=2,outputs=1):
    graph=np.zeros((nodes,nodes),dtype=np.float32)
    for node in range(nodes-1): graph[node,node+1]=1
    membership=np.zeros((nodes,2 if nodes>1 else 1),dtype=np.float32)
    for region in range(membership.shape[1]): membership[np.arange(nodes)%membership.shape[1]==region,region]=1
    membership/=membership.sum(0)
    return DatasetMetadata('tiny',nodes,features,outputs,adjacency=graph,input_layout=dict(kind='window'),
        static_data=dict(region_membership=membership,original_reconstruction=graph,
            regional_reconstruction=membership.T@graph@membership))


def corrected_old_hierarchy(layer,source,supports,regional_supports,membership,mapping):
    # Run independent samples through the old GCN, correcting only Gram axes,
    # device/dtype casts and the formerly unusable one-support/zero-hop cases.
    output=[];regional_output=[];global_output=[]
    for batch in range(len(source)):
        values=source[batch:batch+1]
        regional=membership.T@values
        global_=mapping.T@regional
        vectors=global_[0].permute(1,0,2).flatten(1).detach().numpy()
        graph=np.maximum(vectors@vectors.T-.5,0.)
        dynamic=[]
        for current in (graph,graph.T)[:len(supports)]:
            degree=current.sum(-1,keepdims=True)
            norm=np.divide(current,degree,out=np.zeros_like(current),where=degree>0)
            dynamic.append(torch.tensor(norm,dtype=source.dtype).softmax(-1))
        original=layer.gcn(values,list(supports));regional=layer.gcn(regional,list(regional_supports));global_=layer.gcn(global_,dynamic)
        regional=regional+layer.n1*(mapping@global_).relu()
        original=original+layer.n2*(membership@regional).relu()
        regional=regional+layer.n3*(membership.T@original).relu()
        global_=global_+layer.n4*(mapping.T@regional).relu()
        output.append(original);regional_output.append(regional);global_output.append(global_)
    return torch.cat(output),torch.cat(regional_output),torch.cat(global_output)


@pytest.mark.parametrize('case',[
    (2,5,2,1,4,2,{}),(1,4,3,2,9,3,dict(blocks=2,layers=2,kernel_size=3,residual_channels=3,dilation_channels=5)),
    (2,3,1,1,12,2,dict(blocks=4,layers=2,apt_layer=True,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,global_nodes=15)),
    (1,1,1,1,1,1,dict(layers=1,kernel_size=1,global_nodes=1,order=0))],ids=['two-samples','multioutput','defaults','singletons-zero-hop'])
def test_corrected_original_full_predictions_auxiliary_features_and_all_live_gradients(case):
    batch,nodes,features,outputs,history,horizon,changes=case
    torch.set_num_threads(1);torch.manual_seed(17)
    params={**SMALL,**changes};meta=metadata(nodes,features,outputs)
    current=HIEST(meta,history,horizon,**params).double()
    old_module=importlib.import_module('libcity.model.traffic_speed_prediction.HIEST')
    old=old_module.HIEST(dict(params,input_window=history,output_window=horizon*outputs,device=torch.device('cpu')),
        dict(adj_mx=meta.adjacency.copy(),Mor_mx=meta.static_data['region_membership'].copy(),num_nodes=nodes,feature_dim=features,output_dim=1)).double()
    state={key:value for key,value in current.state_dict().items() if key not in ('Mor_mx','supports','supports_r')}
    loaded=old.load_state_dict(state,strict=False);assert not loaded.unexpected_keys
    assert all(key.startswith(('residual_convs.','bn.','bnc.','bns.')) for key in loaded.missing_keys)
    old.supports=list(current.supports);old.supports_r=list(current.supports_r)
    for layer in old.gconv:
        layer.forward=lambda x,s,r,m,layer=layer:corrected_old_hierarchy(layer,x,s,r,current.Mor_mx,m)
        if not params['order']:
            layer.gcn.forward=lambda x,s,module=layer.gcn:F.dropout(module.mlp(x),module.dropout,module.training)
    if nodes==1: current.eval();old.eval()
    x=torch.randn(batch,history,nodes,features,dtype=torch.float64,requires_grad=True)
    other=x.detach().clone().requires_grad_()
    actual,region,global_,_=current.forward_with_aux(dict(X=x))
    expected,old_region,old_global=old(dict(X=other))
    expected=expected[...,-1:].reshape(batch,horizon,outputs,nodes).transpose(-1,-2)
    for a,b in ((actual,expected),(region,old_region),(global_,old_global)): torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
    (actual.square().sum()+.05*region.square().sum()+.03*global_.square().sum()).backward()
    (expected.square().sum()+.05*old_region.square().sum()+.03*old_global.square().sum()).backward()
    for name,parameter in current.named_parameters():
        assert parameter.grad is not None,name
        torch.testing.assert_close(parameter.grad,dict(old.named_parameters())[name].grad,rtol=1e-6,atol=1e-8)
    torch.testing.assert_close(x.grad,other.grad,rtol=1e-6,atol=1e-8)


def test_biconnected_regions_include_articulation_overlap_and_isolates_with_real_ids():
    ids=[71,3,108,19,250,6,400];graph=np.zeros((7,7))
    for a,b in [(0,1),(1,2),(2,0),(2,3),(3,4),(4,2),(4,5)]: graph[a,b]=1
    mapping,groups=region_mapping(graph,ids)
    assert {frozenset(group) for group in groups}=={frozenset([71,3,108]),frozenset([108,19,250]),frozenset([250,6]),frozenset([400])}
    np.testing.assert_allclose(mapping.sum(0),1.)
    assert np.count_nonzero(mapping[2])==2 and np.count_nonzero(mapping[4])==2
    order=[3,0,6,4,2,5,1]
    other,other_groups=region_mapping(graph[order][:,order],[ids[i] for i in order])
    assert groups==other_groups;np.testing.assert_equal(mapping[order],other)


@pytest.mark.parametrize('groups',[[[0],[0]],[[0]],[[0],[999]],[[0,0],[1]],[]])
def test_invalid_explicit_overlapping_regions_are_rejected(groups):
    with pytest.raises(ValueError,match='HIEST'): region_mapping(np.eye(2),[0,1],groups=groups)


@pytest.mark.parametrize('global_nodes',[1,3])
def test_auxiliary_reconstruction_and_cosine_match_independent_per_sample_formulas_and_gradients(global_nodes):
    torch.manual_seed(17)
    regional=torch.randn(2,3,2,2,dtype=torch.float64,requires_grad=True)
    global_=torch.randn(2,3,global_nodes,2,dtype=torch.float64,requires_grad=True)
    mapping=torch.randn(2,global_nodes,dtype=torch.float64,requires_grad=True).softmax(-1)
    membership=torch.tensor([[1.,0.],[0.,.5],[0.,.5]],dtype=torch.float64)
    original=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],dtype=torch.float64)
    target=membership.T@original@membership
    actual=auxiliary_costs(regional,global_,mapping,membership,original,target)
    expected=[[],[],[]]
    for batch in range(2):
        no=membership@regional[batch];nr=mapping@global_[batch]
        for output,values,truth in ((expected[0],no,original),(expected[1],nr,target)):
            logits=torch.stack([torch.stack([(values[:,i]*values[:,j]).sum() for j in range(values.shape[1])]) for i in range(values.shape[1])])
            output.append((F.softplus(logits)-truth*logits).mean())
        vectors=global_[batch].permute(1,0,2).flatten(1)
        cos=[F.cosine_similarity(vectors[i],vectors[j],dim=0,eps=1e-6) for i in range(global_nodes) for j in range(i+1,global_nodes)]
        expected[2].append(torch.stack(cos).mean() if cos else global_[batch].sum()*0)
    expected=tuple(torch.stack(value) for value in expected)
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    first=torch.autograd.grad(sum(value.sum() for value in actual),(regional,global_,mapping),retain_graph=True)
    second=torch.autograd.grad(sum(value.sum() for value in expected),(regional,global_,mapping))
    for a,b in zip(first,second): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
    zeros=auxiliary_costs(torch.zeros_like(regional),torch.zeros_like(global_),mapping,membership,original,target)
    assert all(torch.isfinite(value).all() for value in zeros)


def configuration(root,name='run'):
    return dict(model=dict(name='HIEST',params={**SMALL,'dropout':.2}),
        data=dict(dataset='HierarchyTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),input_window=4,output_window=2,pad_training=False),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
            scheduler=dict(name='plateau',interval='metric',monitor='val_loss',factor=.5,patience=0)),
        evaluation=dict(batch_size=4),runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def bundle(cfg): return prepare_hiest(cfg['data'],cfg['training'],cfg['evaluation'],17)


def test_data_identity_probability_targets_raw_cache_reuse_and_future_isolation(tmp_path):
    folder=hiest_fixture(tmp_path/'raw');cfg=load_config(config=configuration(tmp_path));first=bundle(cfg)
    static=first.metadata.static_data;assert static['region_membership'].shape==(7,4)
    np.testing.assert_allclose(static['regional_reconstruction'],static['region_membership'].T@static['original_reconstruction']@static['region_membership'],atol=1e-7)
    assert not static['region_membership'].flags.writeable and 400 in first.metadata.info['node_ids']
    sample=first.val.dataset[0];snapshot=deepcopy(vars(first.scaler))
    cfg['data']['graph']['region_groups']=[[71,3,108],[108,19,250,6],[400]]
    changed=bundle(cfg)
    assert changed.split_summary['cache']['status']=='hit' and changed.metadata.static_data['region_membership'].shape==(7,3)
    cfg['data']['graph']['region_groups']=None
    path=folder/'HierarchyTiny.dyna';frame=pd.read_csv(path);future=frame['time']>=sorted(frame['time'].unique())[36]
    frame.loc[future,['value_0','value_1']]+=1000;frame.to_csv(path,index=False)
    changed=bundle(cfg);assert vars(changed.scaler)==snapshot
    torch.testing.assert_close(sample['X'],changed.val.dataset[0]['X'],rtol=0,atol=0)
    assert not torch.equal(sample['y_raw'],changed.val.dataset[0]['y_raw'])


def test_weighted_reconstruction_requires_probabilities_or_explicit_binary_targets(tmp_path):
    folder=hiest_fixture(tmp_path/'raw');path=folder/'HierarchyTiny.rel';frame=pd.read_csv(path);frame['distance']=2.;frame.to_csv(path,index=False)
    cfg=configuration(tmp_path);cfg['data']['graph']=dict(name='weighted',reconstruction='weights');cfg=load_config(config=cfg)
    with pytest.raises(ValueError,match='probabilities'): bundle(cfg)
    cfg['data']['graph']['reconstruction']='binary'
    assert bundle(cfg).metadata.static_data['original_reconstruction'].max()==1


@pytest.mark.parametrize('kind',['doubletransition','transition','symnadj','normlap','scalap','identity'])
def test_all_support_families_and_inference_avoid_last_auxiliary_graph(tmp_path,kind,monkeypatch):
    hiest_fixture(tmp_path/'raw');cfg=configuration(tmp_path);cfg['model']['params']['adjtype']=kind
    context=make_context(cfg)
    try:
        context.model.eval();batch=next(iter(context.bundle.val));inputs=context.loss_fn.inputs(batch)
        prediction=context.model(inputs)
        torch.testing.assert_close(prediction,context.model.forward_with_aux(inputs)[0],rtol=0,atol=0)
        auxiliary_step(context.loss_fn)(context.model,batch,None).loss.backward()
        assert all(parameter.grad is not None for parameter in context.model.parameters())
        def forbidden(*args,**kwargs): raise AssertionError('auxiliary graph executed')
        monkeypatch.setattr(context.model.gconv[-1],'forward',forbidden)
        assert context.model(inputs).shape==(4,2,7,2)
        with pytest.raises(AssertionError,match='auxiliary'): context.model.forward_with_aux(inputs)
    finally: context.tracker.close()


def equal(left,right):
    if isinstance(left,torch.Tensor): torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        for a,b in zip(left,right,strict=True): equal(a,b)
    else: assert left==right


@pytest.mark.parametrize('scaler',['standard','minmax01'])
def test_actual_auxiliary_dropout_batchnorm_train_raw_evaluate_and_exact_resume(tmp_path,scaler):
    hiest_fixture(tmp_path/'raw')
    def config(name):
        cfg=configuration(tmp_path,name);cfg['data']['scaler']=dict(name=scaler);return cfg
    full=run_experiment('train',config=config('full'))
    evaluated=run_experiment('evaluate',config=config('eval'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as source:
        prediction,truth=source['prediction'],source['truth'];valid=np.isfinite(truth)&(truth!=0)
        assert prediction.shape==(11,2,7,2)
        assert evaluated.metrics['loss']==pytest.approx(np.abs(prediction[valid]-truth[valid]).mean(),rel=2e-6)
    assert not (tmp_path/'runs/eval/checkpoints').exists() and not (tmp_path/'runs/eval/metrics.jsonl').exists()
    partial=make_context(config('partial'));before=partial.model.Mrg.detach().clone()
    try:
        train_loop(partial,auxiliary_step(partial.loss_fn),until_epoch=1)
        assert not torch.equal(before,partial.model.Mrg)
    finally: partial.tracker.close()
    resumed=run_experiment('train',config=config('resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual=[torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'): equal(expected[key],actual[key])


def test_global_graph_and_predictions_are_independent_of_other_samples(tmp_path):
    hiest_fixture(tmp_path/'raw');context=make_context(configuration(tmp_path,'independent'))
    try:
        context.model.eval();batch=next(iter(context.bundle.val));x=batch['X'].double()
        context.model.double();current=context.model(dict(X=x))
        separate=torch.cat([context.model(dict(X=x[i:i+1])) for i in range(len(x))])
        torch.testing.assert_close(current,separate,rtol=1e-11,atol=1e-11)
        changed=x.clone();changed[1:]+=100
        torch.testing.assert_close(current[:1],context.model(dict(X=changed))[:1],rtol=0,atol=0)
        values=torch.randn(3,4,2,5,dtype=torch.float64);all_graphs=global_supports(values,2)
        torch.testing.assert_close(all_graphs,torch.cat([global_supports(values[i:i+1],2) for i in range(3)],dim=1),rtol=0,atol=0)
    finally: context.tracker.close()


def test_missing_history_is_imputed_without_losing_original_nan_or_zero_targets(tmp_path):
    folder=hiest_fixture(tmp_path/'raw');path=folder/'HierarchyTiny.dyna';frame=pd.read_csv(path)
    times=sorted(frame['time'].unique())
    frame.loc[frame['time'].isin([times[35],times[36]])&(frame['entity_id']==71),'value_0']=float('nan')
    frame.loc[(frame['time']==times[36])&(frame['entity_id']==3),'value_0']=0.;frame.to_csv(path,index=False)
    cfg=load_config(config=configuration(tmp_path));data=bundle(cfg);sample=data.val.dataset[0]
    assert torch.isfinite(sample['X']).all() and torch.isnan(sample['y_raw']).any() and torch.any(sample['y_raw']==0)
    assert torch.all(sample['y'][sample['y_raw']==0]!=0)
    model=HIEST(data.metadata,4,2,**SMALL).eval()
    assert torch.isfinite(model({name:value[None] for name,value in sample.items()})).all()


def test_physical_node_order_changes_keep_regions_and_predictions_aligned(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(17)
    folder=hiest_fixture(tmp_path/'raw');cfg=load_config(config=configuration(tmp_path));first=bundle(cfg)
    model=HIEST(first.metadata,4,2,**SMALL).eval()
    path=folder/'HierarchyTiny.geo';geo=pd.read_csv(path);geo.iloc[[3,6,0,5,1,4,2]].to_csv(path,index=False)
    changed=bundle(cfg);other=HIEST(changed.metadata,4,2,**SMALL).eval()
    omitted=('Mor_mx','supports','supports_r')
    loaded=other.load_state_dict({key:value for key,value in model.state_dict().items() if key not in omitted},strict=False)
    assert set(loaded.missing_keys)==set(omitted) and not loaded.unexpected_keys
    ids=list(first.metadata.info['node_ids']);order=[ids.index(value) for value in changed.metadata.info['node_ids']]
    assert first.metadata.input_layout['hierarchy']['region_node_ids']==changed.metadata.input_layout['hierarchy']['region_node_ids']
    a=first.val.dataset[0]['X'][None];b=changed.val.dataset[0]['X'][None]
    torch.testing.assert_close(a[:,:,order],b,rtol=0,atol=0)
    torch.testing.assert_close(model(dict(X=a))[:,:,order],other(dict(X=b)),rtol=2e-6,atol=1e-7)


@pytest.mark.parametrize('low_precision',[False,True])
def test_auxiliary_valid_count_weighting_and_microbatch_gradients_include_exact_large_counts(low_precision):
    torch.manual_seed(17)
    nodes=130;membership=np.zeros((nodes,2),dtype=np.float32)
    membership[np.arange(nodes),np.arange(nodes)%2]=1./65
    original=np.eye(nodes,dtype=np.float32)
    meta=DatasetMetadata('probe',nodes,1,1,static_data=dict(region_membership=membership,
        original_reconstruction=original,regional_reconstruction=membership.T@original@membership))
    loss=HIESTLoss(SimpleNamespace(metadata=meta,scaler=NoneScaler()),dict(name='masked_mae',mask_value=0.,
        original_weight=.1,regional_weight=.2,orthogonal_weight=.3))
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prediction=torch.nn.Parameter(torch.randn(3,2,nodes,1,dtype=torch.float64))
            self.regional=torch.nn.Parameter(torch.randn(3,2,2,2,dtype=torch.float64))
            self.global_=torch.nn.Parameter(torch.randn(3,2,3,2,dtype=torch.float64))
            self.mapping=torch.nn.Parameter(torch.randn(2,3,dtype=torch.float64))
        def forward_with_aux(self,batch):
            index=batch['X'][:,0,0,0].long();dtype=torch.bfloat16 if low_precision else torch.float64
            return (self.prediction[index].to(dtype),self.regional[index].to(dtype),self.global_[index].to(dtype),self.mapping.softmax(-1).to(dtype))
    model=Probe();step=auxiliary_step(loss)
    raw=torch.ones(3,2,nodes,1,dtype=torch.float64);raw[0].flatten()[:3]=0;raw[1].flatten()[:1]=0;raw[2]=float('nan')
    batch=dict(X=torch.arange(3,dtype=torch.float64)[:,None,None,None].expand(-1,1,nodes,1),y=raw,y_raw=raw)
    complete=step(model,batch,None);assert complete.weight==516
    prediction,regional,global_,mapping=model.forward_with_aux(batch)
    main=loss(prediction,raw,raw)
    dtype=torch.float32 if low_precision else torch.float64
    constants=loss.constants(regional.to(dtype))
    costs=auxiliary_costs(regional,global_,mapping,*constants)
    aux=sum(weight*value for weight,value in zip(loss.weights,costs))
    expected=main.loss+(aux*torch.tensor([257.,259.,0.],dtype=aux.dtype)).sum()/516
    torch.testing.assert_close(complete.loss,expected,rtol=0,atol=0)
    if not low_precision:
        micros=[step(model,{key:value[i:i+1] for key,value in batch.items()},None) for i in range(3)]
        combined=sum(value.loss*value.weight for value in micros)/516
        torch.testing.assert_close(complete.loss,combined,rtol=1e-12,atol=1e-12)
        first=torch.autograd.grad(complete.loss,tuple(model.parameters()),retain_graph=True)
        second=torch.autograd.grad(combined,tuple(model.parameters()))
        for a,b in zip(first,second): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize('singleton',[False,True])
def test_cpu_amp_real_tail_and_short_auto_layers_global_singleton(tmp_path,singleton):
    if singleton: atomic_fixture(tmp_path/'raw',name='HierarchyTiny',steps=60,nodes=1)
    else: hiest_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,'amp');cfg['training'].update(epochs=1,amp=True)
    if singleton:
        cfg['data']['input_window']=1;cfg['model']['params'].update(apt_layer=True,global_nodes=1)
        cfg['training']['batch_size']=11  # 34 windows: a real one-sample final batch.
    result=run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


def test_windows_spawn_transports_hierarchy_inputs(tmp_path):
    hiest_fixture(tmp_path/'raw');cfg=configuration(tmp_path,'spawn');cfg['training']['num_workers']=1
    context=make_context(cfg)
    try:
        assert next(iter(context.bundle.train))['X'].shape==(5,4,7,2)
    finally: context.tracker.close()


@pytest.mark.parametrize('change',[
    dict(model=dict(params=dict(blocks=0))),dict(model=dict(params=dict(layers=0))),dict(model=dict(params=dict(global_nodes=0))),
    dict(model=dict(params=dict(kernel_size=0))),dict(model=dict(params=dict(order=-1))),dict(model=dict(params=dict(dropout=1.))),
    dict(model=dict(params=dict(apt_layer='yes'))),dict(model=dict(params=dict(adjtype='unknown'))),dict(model=dict(params=dict(n1=-1.))),
    dict(data=dict(graph=dict(region_threshold=-1))),dict(data=dict(graph=dict(region_groups=[[]]))),dict(data=dict(graph=dict(reconstruction='bad'))),
    dict(training=dict(loss=dict(original_weight=-1))),dict(training=dict(loss=dict(orthogonal_weight=False)))])
def test_invalid_hiest_configuration_fails_before_data_reads(change):
    with pytest.raises((ValueError,TypeError)):
        load_config(config=merge_config(dict(model=dict(name='HIEST'),data=dict(dataset='not-read')),change))
