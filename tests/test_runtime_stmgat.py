"""Native directed attention, old temporal reference and full STMGAT lifecycle."""
import ast
from copy import deepcopy
import importlib.abc
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from libcity.config.runtime import load_config,merge_config
from libcity.data.bundle import DatasetMetadata
from libcity.executor.train import train_loop
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.traffic_speed_prediction.runtime_stmgat import STMGAT,EdgeAttention,attention_edges
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,stmgat_fixture
from runtime_helpers import make_context

pytestmark=pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')
SMALL=dict(blocks=1,layers=2,run_gconv=True,residual_channels=4,dilation_channels=4,skip_channels=6,end_channels=8,
    kernel_size=2,heads=3,feat_drop=0.,attn_drop=0.,negative_slope=.2)


def metadata(nodes=4,features=2,outputs=1,graph=None):
    if graph is None:
        graph=np.zeros((nodes,nodes),dtype=np.float32)
        for i in range(nodes-1): graph[i,i+1]=i+1
    return DatasetMetadata('tiny',nodes,features,outputs,adjacency=graph,input_layout=dict(kind='window'))


class ReferenceAttention(nn.Module):
    """Independent incoming-neighborhood formula; DGL is never imported."""
    def __init__(self,inputs,outputs,heads,feat_drop=0.,attn_drop=0.,negative_slope=.2,**kwargs):
        super().__init__();assert inputs==outputs
        self.width,self.heads=outputs,heads;self.slope=negative_slope
        self.fc=nn.Linear(inputs,heads*outputs,bias=False)
        self.attn_l=nn.Parameter(torch.randn(1,heads,outputs));self.attn_r=nn.Parameter(torch.randn(1,heads,outputs))
        self.bias=nn.Parameter(torch.zeros(heads*outputs))
        self.feat_drop=nn.Dropout(feat_drop);self.attn_drop=nn.Dropout(attn_drop)

    def forward(self,graph,source):
        nodes=len(graph);batch=len(source)//nodes
        values=self.fc(self.feat_drop(source)).reshape(batch,nodes,self.heads,self.width)
        outputs=[]
        for sample in values:
            result=[]
            for node in range(nodes):
                incoming=torch.tensor([i for i in range(nodes) if i==node or graph[i,node]>0],device=source.device)
                heads=[]
                for head in range(self.heads):
                    left=sample[incoming,head]@self.attn_l[0,head]
                    right=sample[node,head]@self.attn_r[0,head]
                    attention=self.attn_drop(F.leaky_relu(left+right,self.slope).softmax(0))
                    heads.append((attention[:,None]*sample[incoming,head]).sum(0))
                result.append(F.elu(torch.stack(heads)+self.bias.reshape(self.heads,self.width)))
            outputs.append(torch.stack(result))
        return torch.stack(outputs).reshape(batch*nodes,self.heads,self.width)


@pytest.mark.parametrize('graph',[
    np.zeros((1,1)),np.array([[0,2,0],[0,0,0],[3,1,0]]),np.array([[1,2],[4,5]])])
def test_edge_attention_matches_independent_incoming_formulas_and_all_gradients(graph):
    torch.manual_seed(17);layer=EdgeAttention(3,2,0.,0.,.2).double()
    reference=ReferenceAttention(3,3,2).double();reference.load_state_dict(layer.state_dict())
    source=torch.randn(2,len(graph),3,dtype=torch.float64,requires_grad=True);other=source.detach().clone().requires_grad_()
    actual=layer(source,attention_edges(graph,len(graph)))
    expected=reference(graph,other.flatten(0,1)).mean(1).reshape_as(actual)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    actual.square().sum().backward();expected.square().sum().backward()
    torch.testing.assert_close(source.grad,other.grad,rtol=1e-10,atol=1e-12)
    for name,value in layer.named_parameters():
        torch.testing.assert_close(value.grad,dict(reference.named_parameters())[name].grad,rtol=1e-10,atol=1e-12)


def old_temporal_reference():
    path=Path(__file__).resolve().parents[1]/'libcity/model/traffic_speed_prediction/STMGAT.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[node for node in tree.body if isinstance(node,ast.ClassDef)]
    namespace=dict(torch=torch,nn=nn,F=F,Conv1d=nn.Conv2d,Conv2d=nn.Conv2d,ModuleList=nn.ModuleList,
        BatchNorm2d=nn.BatchNorm2d,GATConv=ReferenceAttention,getLogger=logging.getLogger,
        AbstractTrafficStateModel=AbstractTrafficStateModel,dgl=SimpleNamespace(batch=lambda graphs:graphs[0]))
    exec(compile(tree,str(path),'exec'),namespace)
    result=namespace['STMGAT'];result._get_adj=lambda self:self.data_feature['adj_mx']
    return result


PROFILES=[(2,4,2,1,4,2,{}),(2,4,3,2,9,3,dict(blocks=2,layers=2)),
    (2,3,1,1,12,2,dict(blocks=4,layers=2,residual_channels=40,dilation_channels=40,skip_channels=320,end_channels=640,heads=8)),
    (2,3,2,2,5,3,dict(run_gconv=False,residual_channels=3,dilation_channels=5)),
    (1,1,1,2,1,2,dict(layers=1,kernel_size=1))]


@pytest.mark.parametrize('case',PROFILES,ids=['two-layer','long-history-multioutput','defaults','temporal-only','singleton-one-layer'])
def test_original_temporal_network_and_independent_gat_reference_all_live_gradients(case):
    batch,nodes,features,outputs,history,horizon,changes=case
    torch.set_num_threads(1);torch.manual_seed(17);params={**SMALL,**changes}
    meta=metadata(nodes,features,outputs);current=STMGAT(meta,history,horizon,**params).double()
    old=old_temporal_reference()(dict(params,input_window=history,output_window=horizon*outputs,device='cpu'),
        dict(num_nodes=nodes,feature_dim=features,output_dim=1,adj_mx=meta.adjacency)).double()
    for index,layer in enumerate(current.gat_layers):
        # The old constructor ignored histories longer than its receptive field.
        width=layer.width
        old.gat_layers[index]=ReferenceAttention(width,width,params['heads']).double()
        old.gat_layers1[index]=ReferenceAttention(width,width,params['heads']).double()
    loaded=old.load_state_dict({key:value for key,value in current.state_dict().items() if key!='edge_index'},strict=False)
    assert not loaded.unexpected_keys
    assert all(name.startswith(('residual_convs.','bn.','gat_layers.','gat_layers1.')) for name in loaded.missing_keys)
    if nodes==1: current.eval();old.eval()
    source=torch.randn(batch,history,nodes,features,dtype=torch.float64,requires_grad=True);other=source.detach().clone().requires_grad_()
    actual=current(dict(X=source));expected=old(dict(X=other))[...,-1:]
    expected=expected.reshape(batch,horizon,outputs,nodes).transpose(-1,-2)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-10)
    actual.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(source.grad,other.grad,rtol=1e-7,atol=1e-9)
    for name,value in current.named_parameters():
        assert value.grad is not None,name
        torch.testing.assert_close(value.grad,dict(old.named_parameters())[name].grad,rtol=2e-6,atol=1e-9)
    for name,value in current.named_buffers():
        if name!='edge_index': torch.testing.assert_close(value,old.state_dict()[name],rtol=1e-10,atol=1e-10)


def configuration(root,name='run'):
    return dict(model=dict(name='STMGAT',params={**SMALL,'feat_drop':.2,'attn_drop':.2}),
        data=dict(dataset='GATTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),input_window=4,output_window=2,pad_training=False),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
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


@pytest.mark.parametrize('scaler,graph',[('standard',True),('minmax01',False)])
def test_actual_dropout_batchnorm_train_raw_mse_evaluate_and_exact_resume(tmp_path,scaler,graph):
    stmgat_fixture(tmp_path/'raw')
    def config(name):
        cfg=configuration(tmp_path,name);cfg['data']['scaler']=dict(name=scaler)
        cfg['model']['params']['run_gconv']=graph
        if not graph: cfg['data']['graph']=dict(name='none')
        return cfg
    full=run_experiment('train',config=config('full'))
    evaluated=run_experiment('evaluate',config=config('evaluated'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as result:
        prediction,truth=result['prediction'],result['truth'];valid=np.isfinite(truth)
        assert prediction.shape==(11,2,4,2)
        assert evaluated.metrics['loss']==pytest.approx(np.square(prediction[valid]-truth[valid]).mean(),rel=2e-6)
    assert not (tmp_path/'runs/evaluated/checkpoints').exists() and not (tmp_path/'runs/evaluated/metrics.jsonl').exists()
    partial=make_context(config('partial'))
    try: train_loop(partial,until_epoch=1)
    finally: partial.tracker.close()
    resumed=run_experiment('train',config=config('resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual=[torch.load(result.artifacts['last'],weights_only=True) for result in (full,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'): equal(expected[key],actual[key])


def test_graph_identity_single_self_loop_weight_magnitude_and_inference_independence():
    torch.set_num_threads(1);torch.manual_seed(17)
    graph=np.array([[2,3,0],[0,0,4],[0,0,0]],dtype=np.float32);snapshot=graph.copy()
    meta=metadata(3,2,2,graph);model=STMGAT(meta,5,3,**SMALL).double().eval()
    assert not any(hasattr(model,name) for name in ('device','logger','scaler','_scaler'))
    assert model.edge_index.dtype==torch.long and model.edge_index.shape==(2,5)
    np.testing.assert_array_equal(graph,snapshot)
    graph[:]=99;assert not np.array_equal(meta.adjacency,graph)
    altered=STMGAT(metadata(3,2,2,snapshot*5),5,3,**SMALL).double().eval()
    altered.load_state_dict(model.state_dict())
    x=torch.randn(3,5,3,2,dtype=torch.float64,requires_grad=True);before=torch.get_rng_state().clone()
    prediction=model(dict(X=x,y=torch.tensor(float('nan'))))
    torch.testing.assert_close(prediction,altered(dict(X=x)),rtol=0,atol=0)
    torch.testing.assert_close(prediction,torch.cat([model(dict(X=x[i:i+1])) for i in range(3)]),rtol=1e-11,atol=1e-11)
    changed=x.detach().clone();changed[1:]+=100
    torch.testing.assert_close(prediction[:1],model(dict(X=changed))[:1],rtol=0,atol=0)
    assert torch.equal(before,torch.get_rng_state())
    prediction[0].sum().backward();assert torch.count_nonzero(x.grad[1:])==0
    order=[2,0,1];other=STMGAT(metadata(3,2,2,snapshot[order][:,order]),5,3,**SMALL).double().eval()
    loaded=other.load_state_dict({key:value for key,value in model.state_dict().items() if key!='edge_index'},strict=False)
    assert loaded.missing_keys==['edge_index'] and not loaded.unexpected_keys
    torch.testing.assert_close(prediction[:,:,order],other(dict(X=x[:,:,order])),rtol=1e-10,atol=1e-10)


def test_stable_softmax_is_local_to_each_incoming_neighborhood():
    layer=EdgeAttention(1,1,0.,0.,.2).double()
    with torch.no_grad(): layer.fc.weight.fill_(1);layer.attn_l.fill_(1);layer.attn_r.zero_();layer.bias.zero_()
    graph=np.zeros((4,4));graph[0,1]=1;graph[2,3]=1
    source=torch.tensor([[[1.],[2.],[1000.],[1001.]]],dtype=torch.float64,requires_grad=True)
    output=layer(source,attention_edges(graph,4))
    expected=(torch.tensor([1.,2.],dtype=torch.float64).softmax(0)*torch.tensor([1.,2.],dtype=torch.float64)).sum()
    torch.testing.assert_close(output[0,1,0],expected,rtol=1e-12,atol=1e-12)
    assert torch.isfinite(output).all();output.sum().backward();assert torch.isfinite(source.grad).all()


def test_history_imputation_raw_targets_cache_and_training_prefix_isolation(tmp_path):
    folder=stmgat_fixture(tmp_path/'raw');path=folder/'GATTiny.dyna';frame=pd.read_csv(path);times=sorted(frame['time'].unique())
    frame.loc[(frame['entity_id']==71)&frame['time'].isin([times[35],times[36]]),'value_0']=float('nan')
    frame.loc[(frame['entity_id']==3)&(frame['time']==times[36]),'value_0']=0.;frame.to_csv(path,index=False)
    context=make_context(configuration(tmp_path,'data'))
    try:
        first=context.bundle;sample=first.val.dataset[0];scaler=deepcopy(vars(first.scaler))
        assert torch.isfinite(sample['X']).all() and torch.isnan(sample['y_raw']).any() and torch.any(sample['y_raw']==0)
        assert first.metadata.input_layout['input_missing_observations']=='raw_zero_before_scaling'
        frame.loc[frame['time']>=times[36],['value_0','value_1']]+=1000;frame.to_csv(path,index=False)
        changed=make_context(configuration(tmp_path,'changed'))
        try:
            assert vars(changed.bundle.scaler)==scaler
            torch.testing.assert_close(sample['X'],changed.bundle.val.dataset[0]['X'],rtol=0,atol=0)
            assert not torch.equal(sample['y_raw'],changed.bundle.val.dataset[0]['y_raw'])
        finally: changed.tracker.close()
        warmed=make_context(configuration(tmp_path,'warm'))
        try: assert warmed.bundle.split_summary['cache']['status']=='hit'
        finally: warmed.tracker.close()
        prediction=context.model({key:value[None] for key,value in sample.items()})
        objective=context.loss_fn(prediction,sample['y'][None],sample['y_raw'][None])
        assert torch.isfinite(objective.loss) and objective.weight==int(torch.isfinite(sample['y_raw']).sum())
        restored=context.bundle.scaler.inverse_transform(prediction)[0];valid=torch.isfinite(sample['y_raw'])
        torch.testing.assert_close(objective.loss,(restored[valid]-sample['y_raw'][valid]).square().mean())
    finally: context.tracker.close()


@pytest.mark.parametrize('singleton',[False,True])
def test_cpu_amp_real_singleton_tail_without_dgl(tmp_path,singleton,monkeypatch):
    class DenyDGL(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,*args):
            if fullname=='dgl' or fullname.startswith('dgl.'): raise AssertionError('DGL imported')
    monkeypatch.setattr(sys,'meta_path',[DenyDGL(),*sys.meta_path])
    if singleton: atomic_fixture(tmp_path/'raw',name='GATTiny',steps=60,nodes=1)
    else: stmgat_fixture(tmp_path/'raw')
    cfg=configuration(tmp_path,'amp');cfg['training'].update(epochs=1,amp=True,batch_size=11)
    if singleton: cfg['data']['input_window']=1;cfg['model']['params']['kernel_size']=1
    result=run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()
    state=torch.load(result.artifacts['last'],weights_only=True)
    if singleton: assert state['model']['bn.0.num_batches_tracked']==3


def test_windows_spawn_and_no_rel_temporal_only(tmp_path):
    folder=stmgat_fixture(tmp_path/'raw');(folder/'GATTiny.rel').unlink()
    cfg=configuration(tmp_path,'spawn');cfg['training']['num_workers']=1
    cfg['data']['graph']=dict(name='none');cfg['model']['params']['run_gconv']=False
    context=make_context(cfg)
    try:
        assert next(iter(context.bundle.train))['X'].shape==(5,4,4,3)
        assert context.model.edge_index.numel()==0
        assert not context.model.gat_layers and not context.model.gat_layers1
        assert context.model.residual_convs
    finally: context.tracker.close()


@pytest.mark.parametrize('change',[
    dict(model=dict(params=dict(blocks=0))),dict(model=dict(params=dict(layers=0))),dict(model=dict(params=dict(kernel_size=0))),
    dict(model=dict(params=dict(heads=0))),dict(model=dict(params=dict(residual_channels=3))),dict(model=dict(params=dict(dilation_channels=3))),
    dict(model=dict(params=dict(feat_drop=1.))),dict(model=dict(params=dict(attn_drop=-.1))),dict(model=dict(params=dict(negative_slope=-1.))),
    dict(model=dict(params=dict(run_gconv='yes'))),dict(model=dict(params=dict(dropout=.3))),
    dict(data=dict(sampling=dict(name='cpt',closeness=1,period=0,trend=0)))])
def test_invalid_stmgat_configuration_rejected_before_data(change):
    with pytest.raises((ValueError,TypeError)):
        load_config(config=merge_config(dict(model=dict(name='STMGAT'),data=dict(dataset='not-read')),change))


@pytest.mark.parametrize('graph',[np.full((2,2),np.nan),np.array([[0,-1],[0,0]]),np.ones((3,2))])
def test_invalid_graph_rejected(graph):
    with pytest.raises(ValueError,match='STMGAT'): attention_edges(graph,2)
