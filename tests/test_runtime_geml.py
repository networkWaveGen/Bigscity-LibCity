"""Corrected periodic OD recurrence, raw margins and actual shared lifecycle."""
import importlib
from types import MethodType,SimpleNamespace

import numpy as np
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.executor.evaluate import model_inputs
from libcity.executor.geml_step import GEMLLoss,auxiliary_step
from libcity.executor.train import TrainState,train_loop,backward,optimizer_update
from libcity.model.traffic_od_prediction.runtime_geml import GEML,geographic_graph,semantic_graph
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import od_fixture
from libcity.utils.normalization import NoneScaler,StandardScaler,MinMax01Scaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(nodes=3):
    graph = np.arange(nodes*nodes,dtype=float).reshape(nodes,nodes)
    np.fill_diagonal(graph,0.)
    return DatasetMetadata('tiny',nodes,1,1,graph,input_layout=dict(kind='od_window'))


def configuration(root,name):
    return dict(model=dict(name='GEML',params=dict(embed_dim=4,p_interval=2)),
        data=dict(dataset='ODTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=3,pad_training=False),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
            scheduler=dict(name='plateau',interval='metric',monitor='val_loss',factor=.5,patience=0)),
        evaluation=dict(batch_size=4),runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def corrected_memory(self,values):
    # A time-indexed reference reads t-P; no state aliases or batch-row indexing.
    hidden,cells = [],[]
    empty = values.new_zeros(values.shape[1],self.hidden_dim)
    for time,value in enumerate(values):
        previous = time-self.p_interval
        source = torch.cat((value,hidden[previous] if previous>=0 else empty),-1)
        memory = cells[previous] if previous>=0 else empty
        cells.append(torch.tanh(self.f_gate(source)*memory+self.i_gate(source)*self.g_gate(source)))
        hidden.append(self.o_gate(source)*cells[-1])
    return hidden[-1]


@pytest.mark.parametrize('steps,horizon,embedding,period',[(12,2,32,1),(4,3,4,2),(7,2,3,3),(1,3,2,2)])
def test_corrected_original_full_forecast_auxiliary_heads_and_all_gradients(steps,horizon,embedding,period):
    legacy = importlib.import_module('libcity.model.traffic_od_prediction.GEML')
    torch.manual_seed(17)
    meta = metadata()
    old = legacy.GEML(dict(input_window=steps,output_window=horizon,output_dim=1,embed_dim=embedding,
        p_interval=period,batch_size=2,device=torch.device('cpu')),dict(num_nodes=3,adj_mx=meta.adjacency,scaler=NoneScaler()))
    current = GEML(metadata=meta,input_window=steps,output_window=horizon,embed_dim=embedding,p_interval=period)
    loaded = current.load_state_dict(old.state_dict(),strict=False)
    assert loaded.missing_keys==['geo_adj'] and not loaded.unexpected_keys
    current,old = current.double(),old.double()
    old.geo_adj = current.geo_adj.expand(2,steps,-1,-1)
    old.LSTM.forward = MethodType(corrected_memory,old.LSTM)
    source = torch.rand(2,steps,3,3,1,dtype=torch.float64,requires_grad=True)
    reference = source.detach().clone().requires_grad_()
    expected = old.predict(dict(X=reference))
    actual = current.forward_with_aux(dict(X=source,history_mask=source>0))
    for value,result in zip(actual,expected,strict=True):
        torch.testing.assert_close(value,result,rtol=2e-10,atol=2e-11)
    sum(value.square().mean() for value in actual).backward()
    sum(value.square().mean() for value in expected).backward()
    torch.testing.assert_close(source.grad,reference.grad,rtol=2e-8,atol=2e-10)
    previous = dict(old.named_parameters())
    for name,value in current.named_parameters():
        assert value.grad is not None and previous[name].grad is not None
        torch.testing.assert_close(value.grad,previous[name].grad,rtol=2e-8,atol=3e-10)


def test_geographic_squared_distance_formula_zero_rows_and_large_finite_distances():
    distance = torch.tensor([[0.,2.,3.],[0.,0.,0.],[4.,1.,0.]],dtype=torch.float64)
    expected = np.square(distance.numpy())
    expected /= np.where(expected.sum(1,keepdims=True)>0,expected.sum(1,keepdims=True),1)
    np.fill_diagonal(expected,1.)
    np.testing.assert_allclose(geographic_graph(distance).numpy(),expected,rtol=1e-15,atol=1e-15)
    giant = torch.tensor([[0.,1e300],[0.,0.]],dtype=torch.float64)
    torch.testing.assert_close(geographic_graph(giant),torch.tensor([[1.,1.],[0.,1.]],dtype=torch.float64))


def test_semantic_graph_matches_independent_degree_formula_and_raw_mask_not_normalized_sign():
    mask = torch.tensor([[[False,True,False],[False,False,True],[False,False,False]]])
    adjacency = (mask.numpy()|mask.numpy().transpose(0,2,1)).astype(float)
    degree = adjacency.sum(-1)
    expected = np.empty_like(adjacency)
    for i in range(3):
        for j in range(3):
            expected[0,i,j] = 1. if i==j else degree[0,j]/(.001+sum(adjacency[0,i,k]*degree[0,k] for k in range(3)))
    np.testing.assert_allclose(semantic_graph(mask,torch.float64).numpy(),expected,rtol=1e-14)
    torch.testing.assert_close(semantic_graph(torch.zeros(1,3,3,dtype=torch.bool),torch.float64),torch.eye(3,dtype=torch.float64)[None])


@pytest.mark.parametrize('nodes,steps,period',[(3,4,2),(1,1,3)])
def test_batch_independent_label_free_float64_rollout_keeps_only_static_buffer_and_does_not_call_aux_heads(nodes,steps,period,monkeypatch):
    model = GEML(metadata=metadata(nodes),input_window=steps,output_window=3,embed_dim=3,p_interval=period).double().eval()
    source = torch.randn(3,steps,nodes,nodes,1,dtype=torch.float64)
    mask = torch.rand_like(source)>.4
    inputs = dict(X=source,history_mask=mask)
    state = torch.get_rng_state().clone()
    expected = model(inputs)
    actual = torch.cat([model(dict(X=source[index:index+1],history_mask=mask[index:index+1])) for index in range(3)])
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    monkeypatch.setattr(model.mutiLearning,'forward',lambda *_:pytest.fail('Inference must not calculate auxiliary heads'))
    extra = dict(inputs,y=torch.full_like(expected,float('nan')),future_features=torch.randn(3,3,8),history_features=torch.randn(3,steps,8))
    state = torch.get_rng_state().clone()
    torch.testing.assert_close(model(extra),expected,rtol=0,atol=0)
    assert torch.equal(state,torch.get_rng_state())
    assert set(dict(model.named_buffers()))=={'geo_adj'} and model.geo_adj.shape==(nodes,nodes)
    assert not any(hasattr(model,name) for name in ('semantic_adj','_scaler','logger','device'))


class FixedHeads(torch.nn.Module):
    def __init__(self,shape):
        super().__init__()
        self.od = torch.nn.Parameter(torch.randn(shape,dtype=torch.float64))
        self.origin = torch.nn.Parameter(torch.randn(*shape[:3],1,1,dtype=torch.float64))
        self.destination = torch.nn.Parameter(torch.randn(*shape[:3],1,1,dtype=torch.float64))

    def forward_with_aux(self,inputs):
        return self.od,self.origin,self.destination


@pytest.mark.parametrize('scaler',[NoneScaler(),StandardScaler(10.,2.),MinMax01Scaler(5.,20.)],ids=['none','standard','minmax01'])
@pytest.mark.parametrize('missing',[False,True])
@pytest.mark.parametrize('mask_value',[None,0.])
def test_raw_marginal_objective_and_all_head_gradients_match_direct_cellwise_formula(scaler,missing,mask_value):
    torch.manual_seed(17)
    shape = (2,3,3,3,1)
    target = torch.rand(shape,dtype=torch.float64)*20
    target[0,0,0,0] = 0.
    if missing:
        target[1,2,1,0] = float('nan')
    model = FixedHeads(shape)
    loss = GEMLLoss(SimpleNamespace(scaler=scaler,metadata=metadata()),
        dict(name='masked_mse',mask_value=mask_value,od_weight=.5,origin_weight=.25,destination_weight=.25))
    batch = dict(X=torch.empty(0),y=scaler.transform(target),y_raw=target)
    actual = auxiliary_step(loss)(model,batch,TrainState())
    raw,origins,destinations = [scaler.inverse_transform(value) for value in (model.od,model.origin,model.destination)]
    terms = []
    for b in range(2):
        for h in range(3):
            for origin in range(3):
                for destination in range(3):
                    value = target[b,h,origin,destination,0]
                    if not torch.isfinite(value) or mask_value is not None and value==mask_value:
                        continue
                    term = .5*(raw[b,h,origin,destination,0]-value).square()
                    row,col = target[b,h,origin,:,0],target[b,h,:,destination,0]
                    if torch.isfinite(row).all():
                        term = term+.25*(origins[b,h,origin,0,0]-row.sum()).square()
                    if torch.isfinite(col).all():
                        term = term+.25*(destinations[b,h,destination,0,0]-col.sum()).square()
                    terms.append(term)
    expected = torch.stack(terms).mean()
    assert actual.weight==len(terms)
    torch.testing.assert_close(actual.loss,expected,rtol=1e-13,atol=1e-12)
    first = torch.autograd.grad(actual.loss,tuple(model.parameters()))
    second = torch.autograd.grad(expected,tuple(model.parameters()))
    for a,b in zip(first,second,strict=True):
        torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-12)


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


@pytest.mark.parametrize('scaler',['minmax01','standard'])
def test_actual_od_train_independent_evaluation_auxiliary_updates_and_full_adam_schedule_rng_resume(tmp_path,scaler):
    od_fixture(tmp_path/'raw',features=1)
    def config(name):
        result = configuration(tmp_path,name)
        result['data']['scaler'] = dict(name=scaler)
        return result
    trained = run_experiment('train',config=config('full'))
    evaluated = run_experiment('evaluate',config=config('evaluate'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(3,3,3,1)
        assert evaluated.metrics['loss']==pytest.approx(np.square(archive['prediction']-archive['truth']).mean(),rel=2e-6)
    partial = make_context(config('partial'))
    before = partial.model.mutiLearning.project_in.detach().clone()
    try:
        train_loop(partial,auxiliary_step(partial.loss_fn),until_epoch=1)
        assert not torch.equal(before,partial.model.mutiLearning.project_in)
    finally:
        partial.tracker.close()
    resumed = run_experiment('train',config=config('resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = [torch.load(item.artifacts['last'],weights_only=True) for item in (trained,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])


def test_effective_batch_auxiliary_update_with_missing_cells_matches_full_batch(tmp_path):
    od_fixture(tmp_path/'raw',features=1)
    def config(name):
        result = configuration(tmp_path,name)
        result['training'].update(optimizer=dict(name='sgd',lr=.0001,momentum=.9,weight_decay=.01),
            scheduler=dict(name='step',interval='step',step_size=1,gamma=.9),loss=dict(mask_value=0.))
        return result
    full,micro = make_context(config('full')),make_context(config('micro'))
    try:
        full.model.double();micro.model.double()
        samples = [full.bundle.train.dataset[index] for index in range(3)]
        batch = {key:torch.stack([sample[key] for sample in samples]) for key in samples[0]}
        batch = {key:value.double() if value.is_floating_point() else value for key,value in batch.items()}
        batch['y_raw'][0,0,0,0] = float('nan')
        batch['y_raw'][1,1,1,2] = 0.
        batch['y'] = full.bundle.scaler.transform(batch['y_raw'])
        for context,parts in ((full,[slice(None)]),(micro,[slice(0,1),slice(1,None)])):
            weight = 0
            for part in parts:
                result = auxiliary_step(context.loss_fn)(context.model,{key:value[part] for key,value in batch.items()},context.state)
                backward(context,result.loss*result.weight)
                weight += result.weight
            assert optimizer_update(context,weight)
        for left,right in zip(full.model.parameters(),micro.model.parameters(),strict=True):
            torch.testing.assert_close(left,right,rtol=1e-12,atol=1e-12)
        assert full.state.optimizer_steps==micro.state.optimizer_steps==1
        assert full.optimizer.param_groups[0]['lr']==micro.optimizer.param_groups[0]['lr']
    finally:
        full.tracker.close();micro.tracker.close()


def test_empty_od_targets_skip_all_auxiliary_terms_and_keep_finite_zero_gradient():
    model = FixedHeads((2,3,3,3,1))
    loss = GEMLLoss(SimpleNamespace(scaler=NoneScaler(),metadata=metadata()),
        dict(name='masked_mse',mask_value=None,od_weight=.5,origin_weight=.25,destination_weight=.25))
    batch = dict(X=torch.empty(0),y=torch.full_like(model.od,float('nan')),y_raw=torch.full_like(model.od,float('nan')))
    result = auxiliary_step(loss)(model,batch,TrainState())
    assert result.weight==0 and result.loss.item()==0
    result.loss.backward()
    assert torch.count_nonzero(model.od.grad)==0


@pytest.mark.parametrize('singleton',[False,True])
def test_actual_cpu_amp_with_real_tail_and_single_node_single_history(tmp_path,singleton):
    od_fixture(tmp_path/'raw',steps=50,features=1,nodes=1 if singleton else 3)
    config = configuration(tmp_path,'amp')
    config['training'].update(epochs=1,amp=True)
    config['data']['output_window'] = 1
    if singleton:
        config['data']['input_window'] = 1
    result = run_experiment('train',config=config)
    assert np.isfinite(list(result.metrics.values())).all()


def test_spawn_worker_keeps_boolean_raw_masks(tmp_path):
    od_fixture(tmp_path/'raw',features=1)
    config = configuration(tmp_path,'spawn')
    config['training']['num_workers'] = 1
    context = make_context(config)
    try:
        batch = next(iter(context.bundle.train))
        assert batch['X'].shape==(5,4,3,3,1) and batch['history_mask'].dtype==torch.bool
        assert context.model(model_inputs(batch,1)).shape==(5,3,3,3,1)
    finally:
        context.tracker.close()


@pytest.mark.parametrize('change',[
    dict(data=dict(sampling=dict(name='window'))),dict(data=dict(graph=dict(name='distance'))),
    dict(model=dict(params=dict(p_interval=0))),dict(model=dict(params=dict(embed_dim=0))),
    dict(training=dict(loss=dict(name='mae'))),dict(training=dict(loss=dict(origin_weight=-1.))),
    dict(training=dict(loss=dict(od_weight=0.,origin_weight=0.,destination_weight=0.)))])
def test_invalid_od_configuration_fails_before_data_loading(change):
    from libcity.config.runtime import merge_config
    with pytest.raises(ValueError):
        load_config(config=merge_config(dict(model=dict(name='GEML'),data=dict(dataset='missing')),change))


def test_other_models_cannot_select_od_sampling():
    with pytest.raises(ValueError,match='OD model'):
        load_config(config=dict(model=dict(name='GRU'),data=dict(dataset='missing',sampling=dict(name='od_window'))))


def test_missing_flow_metadata_and_invalid_mask_fail_clearly():
    with pytest.raises(ValueError,match='one flow'):
        GEML(metadata=DatasetMetadata('bad',3,2,1,input_layout=dict(kind='od_window')),input_window=4,output_window=2)
    with pytest.raises(ValueError,match='adjacency'):
        GEML(metadata=DatasetMetadata('bad',3,1,1,input_layout=dict(kind='od_window')),input_window=4,output_window=2)
    model = GEML(metadata=metadata(),input_window=4,output_window=2)
    with pytest.raises(ValueError,match='boolean mask'):
        model(dict(X=torch.zeros(1,4,3,3,1),history_mask=torch.zeros(1,4,3,3,1)))
