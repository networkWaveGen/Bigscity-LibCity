"""Grid identity, column correlations and causal CSTN external observations."""
import importlib
from types import MethodType

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config,merge_config
from libcity.data.bundle import DatasetMetadata
from libcity.data.od_bundle import prepare_od
from libcity.model.traffic_od_prediction.runtime_cstn import CSTN,GCC
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import od_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(height=2,width=3,features=3,known=None,order=None):
    nodes = height*width
    order = np.arange(nodes) if order is None else np.asarray(order)
    columns = [f'external_{index}' for index in range(features)]
    known = columns if known is None else known
    return DatasetMetadata('tiny',nodes,1,1,input_layout=dict(kind='od_window',grid_shape=[height,width],
        grid_positions=np.stack((order//width,order%width),1).tolist(),
        history_features=dict(kind='global_covariates',columns=columns),future_feature_names=known))


def configuration(root,name):
    return dict(model=dict(name='CSTN',params=dict(n_layers=2,c_lt=8)),
        data=dict(dataset='ODTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=3,pad_training=False,features=dict(known_future_columns=['holiday'])),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
            scheduler=dict(name='plateau',interval='metric',monitor='val_loss',factor=.5,patience=0)),
        evaluation=dict(batch_size=4),runtime=dict(device='cpu',seed=17,exp_id=name,output_dir=str(root/'runs')))


def corrected_weather_forward(self,batch):
    # Keep the legacy modules; only fix the spatial broadcast of each MLP
    # channel, rather than repeating the entire 8-channel vector per pixel.
    batch_size = batch['X'].shape[0]
    spatial = self.LSC(batch['X'][...,0]).reshape(batch_size,self.input_window,32,self.height,self.width)
    weather = self.MLP(batch['W'])
    weather = torch.stack([weather for _ in range(self.height*self.width)],-1)
    weather = weather.reshape(batch_size,self.input_window,8,self.height,self.width)
    result = self.OUT(self.GCC(self.TEC(torch.cat((spatial,weather),2))))
    return result.reshape(batch_size,1,self.height,self.width,self.height,self.width,1)


def typed_hidden(self,batch_size,image_size):
    shape = (batch_size,self.hidden_dim,*image_size)
    return self.conv.weight.new_zeros(shape),self.conv.weight.new_zeros(shape)


@pytest.mark.parametrize('height,width,layers,channels,steps,horizon',[
    (2,3,3,75,4,2),(1,2,1,7,3,3),(2,1,2,5,1,2),(1,1,3,3,2,1)])
def test_repaired_original_full_network_forecast_and_all_gradients(height,width,layers,channels,steps,horizon):
    legacy = importlib.import_module('libcity.model.traffic_od_prediction.CSTN')
    torch.manual_seed(17)
    old = legacy.CSTN(dict(input_window=steps,output_window=horizon,n_layers=layers,c_lt=channels,
        output_dim=1,batch_size=2),dict(len_row=height,len_column=width,scaler=NoneScaler()))
    current = CSTN(metadata=metadata(height,width,29),input_window=steps,output_window=horizon,n_layers=layers,c_lt=channels)
    loaded = current.load_state_dict(old.state_dict(),strict=False)
    assert set(loaded.missing_keys)=={'node_to_grid','grid_to_node','future_positions'} and not loaded.unexpected_keys
    current,old = current.double(),old.double()
    old.GCC.Softmax = torch.nn.Softmax(dim=1)
    old.forward = MethodType(corrected_weather_forward,old)
    for cell in old.TEC.ConvLSTM.cell_list:
        cell.init_hidden = MethodType(typed_hidden,cell)
    nodes = height*width
    source = torch.rand(2,steps,nodes,nodes,1,dtype=torch.float64,requires_grad=True)
    reference = source.detach().clone().requires_grad_()
    covariates = torch.randn(2,steps+horizon,29,dtype=torch.float64,requires_grad=True)
    old_covariates = covariates.detach().clone().requires_grad_()
    expected = old.predict(dict(X=reference.reshape(2,steps,height,width,height,width,1),W=old_covariates,
        y=torch.zeros(2,horizon,height,width,height,width,1,dtype=torch.float64)))
    expected = expected.reshape(2,horizon,nodes,nodes,1)
    actual = current(dict(X=source,history_features=covariates[:,:steps],future_features=covariates[:,steps:]))
    torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-11)
    actual.square().mean().backward();expected.square().mean().backward()
    torch.testing.assert_close(source.grad,reference.grad,rtol=2e-8,atol=3e-11)
    torch.testing.assert_close(covariates.grad,old_covariates.grad,rtol=2e-8,atol=3e-11)
    previous = dict(old.named_parameters())
    for name,value in current.named_parameters():
        assert value.grad is not None and previous[name].grad is not None
        torch.testing.assert_close(value.grad,previous[name].grad,rtol=2e-8,atol=3e-11)


@pytest.mark.parametrize('batch',[1,3])
def test_global_correlations_match_independent_column_normalized_weighted_sum(batch):
    source = torch.randn(batch,3,2,3,dtype=torch.float64,requires_grad=True)
    actual = GCC()(source)
    flat = source.flatten(2)
    scores = flat.transpose(1,2)@flat
    columns = []
    for destination in range(6):
        logits = scores[:,:,destination]
        weights = (logits-logits.max(-1,keepdim=True).values).exp()
        weights = weights/weights.sum(-1,keepdim=True)
        columns.append((flat*weights[:,None]).sum(-1))
    expected = torch.cat((torch.stack(columns,-1).reshape_as(source),source),1)
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    first = torch.autograd.grad(actual.square().sum(),source,retain_graph=True)[0]
    second = torch.autograd.grad(expected.square().sum(),source)[0]
    torch.testing.assert_close(first,second,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('features',[0,3])
def test_sample_independence_label_free_float64_and_no_inference_rng(features):
    torch.manual_seed(17)
    model = CSTN(metadata=metadata(2,2,features),input_window=2,output_window=2,n_layers=1,c_lt=5).double().eval()
    source = torch.randn(3,2,4,4,1,dtype=torch.float64,requires_grad=True)
    inputs = dict(X=source)
    if features:
        inputs.update(history_features=torch.randn(3,2,features,dtype=torch.float64),
            future_features=torch.randn(3,2,features,dtype=torch.float64))
    state = torch.get_rng_state().clone()
    expected = model(inputs)
    independent = torch.cat([model({key:value[index:index+1] for key,value in inputs.items()}) for index in range(3)])
    torch.testing.assert_close(independent,expected,rtol=1e-12,atol=1e-12)
    gradient = torch.autograd.grad(expected[0].sum(),source)[0]
    assert torch.count_nonzero(gradient[1:])==0
    dirty = dict(inputs,y=torch.full_like(expected,float('nan')),y_raw=torch.full_like(expected,float('nan')))
    torch.testing.assert_close(model(dirty),expected,rtol=0,atol=0)
    assert torch.equal(state,torch.get_rng_state())
    assert not any(hasattr(model,name) for name in ('_scaler','logger','device'))
    assert (model.MLP is None)==(features==0)


def test_geo_permutation_is_inverted_for_both_od_axes():
    torch.manual_seed(17)
    base = CSTN(metadata=metadata(2,2,0),input_window=3,output_window=2,n_layers=2,c_lt=5).double()
    order = torch.tensor([2,0,3,1])
    reordered = CSTN(metadata=metadata(2,2,0,order=order.numpy()),input_window=3,output_window=2,n_layers=2,c_lt=5).double()
    params = dict(base.named_parameters())
    with torch.no_grad():
        for name,value in reordered.named_parameters():
            value.copy_(params[name])
    source = torch.randn(2,3,4,4,1,dtype=torch.float64)
    expected = base(dict(X=source)).index_select(2,order).index_select(3,order)
    actual = reordered(dict(X=source.index_select(2,order).index_select(3,order)))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_weather_channels_are_constant_over_space_instead_of_interleaved():
    model = CSTN(metadata=metadata(2,3,3,known=[]),input_window=2,output_window=1,n_layers=1,c_lt=4)
    captured = []
    handle = model.TEC.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach()))
    weather = torch.randn(2,2,3)
    try:
        model(dict(X=torch.rand(2,2,6,6,1),history_features=weather))
    finally:
        handle.remove()
    expected = model.MLP(weather)[...,None,None].expand(-1,-1,-1,2,3)
    torch.testing.assert_close(captured[0][:,:,32:],expected,rtol=0,atol=0)


def test_rolling_covariates_advance_only_known_columns_and_persist_unobserved_weather():
    model = CSTN(metadata=metadata(1,2,3,known=['external_2','external_0']),
        input_window=2,output_window=3,n_layers=1,c_lt=4)
    seen = []
    handle = model.MLP.register_forward_pre_hook(lambda _,args:seen.append(args[0].detach().clone()))
    weather = torch.tensor([[[1.,2.,3.],[4.,5.,6.]]])
    future = torch.tensor([[[10.,11.],[12.,13.],[99.,100.]]])
    try:
        model(dict(X=torch.rand(1,2,2,2,1),history_features=weather,future_features=future))
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0],weather)
    torch.testing.assert_close(seen[1],torch.tensor([[[4.,5.,6.],[11.,5.,10.]]]))
    torch.testing.assert_close(seen[2],torch.tensor([[[11.,5.,10.],[13.,5.,12.]]]))


def test_actual_grid_od_cache_ignores_rel_and_future_truth_or_weather_cannot_change_first_validation_forecast(tmp_path):
    folder = od_fixture(tmp_path/'raw',features=1,grid_shape=(2,2))
    (folder/'ODTiny.rel').write_text('invalid unused relation',encoding='utf-8')
    cfg = load_config(config=configuration(tmp_path,'data'))
    prepare = lambda:prepare_od(cfg['data'],cfg['training'],cfg['evaluation'],17)
    bundle = prepare()
    assert bundle.split_summary['cache']['status']=='miss'
    (folder/'ODTiny.rel').write_text('changed unused relation',encoding='utf-8')
    cached = prepare()
    assert cached.split_summary['cache']['status']=='hit'
    assert not cached.metadata.adjacency.any()
    sample = bundle.val.dataset[0]
    source = folder/'ODTiny.gridod'
    frame = pd.read_csv(source)
    times = sorted(frame['time'].unique())
    frame.loc[frame['time'].isin(times[28:]),'flow_0'] += 1000
    frame.to_csv(source,index=False)
    external = pd.read_csv(folder/'ODTiny.ext')
    external.loc[external['time'].isin(times[28:]),'weather'] += 500
    external.to_csv(folder/'ODTiny.ext',index=False)
    changed = prepare()
    assert vars(bundle.scaler)==vars(changed.scaler)
    for key in ('X','history_features','future_features'):
        torch.testing.assert_close(changed.val.dataset[0][key],sample[key],rtol=0,atol=0)
    model = CSTN(metadata=bundle.metadata,input_window=4,output_window=3,n_layers=1,c_lt=4)
    inputs = lambda row:{key:row[key][None] for key in ('X','history_features','future_features')}
    torch.testing.assert_close(model(inputs(sample)),model(inputs(changed.val.dataset[0])),rtol=0,atol=0)
    assert not torch.equal(changed.val.dataset[0]['y_raw'],sample['y_raw'])


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
def test_actual_grid_od_train_independent_evaluation_and_adam_metric_schedule_rng_resume(tmp_path,scaler):
    od_fixture(tmp_path/'raw',features=1,grid_shape=(2,2))
    def config(name):
        result = configuration(tmp_path,name)
        result['data']['scaler'] = dict(name=scaler)
        return result
    trained = run_experiment('train',config=config('full'))
    evaluated = run_experiment('evaluate',config=config('evaluate'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(3,4,4,1)
        assert evaluated.metrics['loss']==pytest.approx(np.square(archive['prediction']-archive['truth']).mean(),rel=2e-6)
    partial = make_context(config('partial'))
    before = partial.model.TEC.ConvLSTM.cell_list[0].conv.weight.detach().clone()
    from libcity.executor.train import train_loop
    try:
        train_loop(partial,until_epoch=1)
        assert not torch.equal(before,partial.model.TEC.ConvLSTM.cell_list[0].conv.weight)
    finally:
        partial.tracker.close()
    resumed = run_experiment('train',config=config('resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = [torch.load(item.artifacts['last'],weights_only=True) for item in (trained,resumed)]
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])


@pytest.mark.parametrize('singleton',[False,True])
def test_actual_cpu_amp_real_tail_and_no_external_single_grid_cell(tmp_path,singleton):
    od_fixture(tmp_path/'raw',steps=50,features=1,grid_shape=(1,1) if singleton else (2,2))
    cfg = configuration(tmp_path,'amp')
    cfg['training'].update(epochs=1,amp=True)
    cfg['data']['output_window'] = 1
    if singleton:
        cfg['data'].update(input_window=1,features=dict(load_external=False,time_in_day=False,known_future_columns=[]))
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


@pytest.mark.parametrize('change',[
    dict(data=dict(sampling=dict(name='window'))),dict(data=dict(graph=dict(name='weighted'))),
    dict(data=dict(graph=dict(symmetric=True))),dict(data=dict(graph=dict(self_loops=True))),
    dict(model=dict(params=dict(n_layers=0))),dict(model=dict(params=dict(c_lt=0)))])
def test_invalid_cstn_configuration_fails_before_loading_data(change):
    with pytest.raises(ValueError):
        load_config(config=merge_config(dict(model=dict(name='CSTN'),data=dict(dataset='missing')),change))


@pytest.mark.parametrize('layout',[
    dict(grid_shape=None),dict(grid_shape=[2,2],grid_positions=[[0,0]]*4),
    dict(grid_positions=[[0,0],[0,1],[1,0],[2,1]]),
    dict(future_feature_names=['not_a_historical_column'])])
def test_invalid_grid_or_covariate_metadata_fails_clearly(layout):
    original = metadata(2,2)
    changed = dict(original.input_layout,**layout)
    with pytest.raises(ValueError,match='CSTN'):
        CSTN(metadata=DatasetMetadata('bad',4,1,1,input_layout=changed),input_window=4,output_window=2)


def test_missing_or_incorrect_covariate_tensors_fail_clearly():
    model = CSTN(metadata=metadata(1,2,3),input_window=2,output_window=2,n_layers=1,c_lt=4)
    source = torch.rand(1,2,2,2,1)
    with pytest.raises(ValueError,match='history_features'):
        model(dict(X=source))
    with pytest.raises(ValueError,match='future_features'):
        model(dict(X=source,history_features=torch.rand(1,2,3)))
    with pytest.raises(ValueError,match='X must'):
        model(dict(X=source[:,0]))
