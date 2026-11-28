"""Dynamic CPT graphs retain the default network while removing fixed axes."""
import importlib
import re

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_flow_prediction.runtime_dgcn import DGCN,DynamicChebyshev,SpatialAttention,graph_support,scaled_laplacian
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,periodic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(counts=(2,1,2),horizon=12,nodes=3,features=1,outputs=2):
    closeness,period,trend = counts
    graph = np.roll(np.eye(nodes,dtype=np.float32),1,axis=1) if nodes>1 else np.zeros((1,1),dtype=np.float32)
    return DatasetMetadata('tiny',nodes,features,outputs,adjacency=graph,info=dict(time_intervals=300),
        input_layout=dict(kind='cpt',length=sum(counts)*horizon,
            sampling=dict(closeness=closeness,period=period,trend=trend)))


def configuration(root,name):
    return dict(data=dict(dataset='Tiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=1,output_window=1,pad_training=False),
        model=dict(name='DGCN',params=dict(c_out=4,K=2,dropout=.2)),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2),evaluation=dict(batch_size=7),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


def native_key(key):
    if key=='h':
        return 'graph_residual'
    if key.startswith('bn.'):
        return key.replace('bn.','input_normalization.',1)
    if re.match(r'^conv[1-4]\.',key):
        return re.sub(r'^conv([1-4])\.',lambda item:f'prediction_heads.{int(item[1])-1}.',key)
    key = key.replace('block1.','blocks.0.').replace('block2.','blocks.1.')
    for before,after in {'TATT_1':'temporal_attention','SATT_3':'coarse_attention','SATT_2':'fine_attention',
            'dynamic_gcn':'graph_conv','LSTM':'graph_recurrence'}.items():
        key = key.replace(before,after)
    return re.sub(r'^(blocks\.\d+)\.bn\.',r'\1.normalization.',key)


def correct_legacy_coarse(module):
    original = module.forward
    def forward(value):
        batch,channels,nodes,length = value.shape
        chunks = torch.stack([value[...,offset:offset+12] for offset in range(0,length,12)],-1)
        packed = chunks.permute(0,1,3,2,4).reshape(batch,channels*12,nodes,length//12)
        # Undo the legacy implementation's interleaving before invoking its
        # unchanged learned projections, multi-head products and sigmoid mean.
        incoming = packed.permute(0,1,3,2).reshape(batch,channels,length,nodes).permute(0,1,3,2)
        return original(incoming)
    module.forward = forward


@pytest.mark.parametrize('channels,training',[(4,False),(8,True),(64,False)])
def test_default_sixty_frame_corrected_original_forecast_all_gradients_and_bn_state(channels,training):
    torch.manual_seed(11)
    meta = metadata()
    current = DGCN(metadata=meta,input_window=12,output_window=12,c_out=channels).double().train(training)
    old_type = importlib.import_module('libcity.model.traffic_flow_prediction.DGCN').DGCN
    old = old_type(dict(c_out=channels,device=torch.device('cpu')),
        dict(num_nodes=3,feature_dim=1,output_dim=2,len_closeness=24,len_period=12,len_trend=24,
            adj_mx=meta.adjacency.copy(),scaler=NoneScaler())).double().train(training)
    old.supports = current.supports.clone()
    for block,modern in zip((old.block1,old.block2),current.blocks):
        correct_legacy_coarse(block.SATT_3)
        block.LSTM.register_forward_pre_hook(lambda _,args:(args[0],tuple(value.to(args[0]) for value in args[1])))
        block.TATT_1.B = torch.where(modern.temporal_attention.mask,0.,-1e13).double()
    incoming = {native_key(key):value for key,value in old.state_dict().items()
        if '.SATT_3.bn.' not in key and '.SATT_2.bn.' not in key}
    loaded = current.load_state_dict(incoming,strict=False)
    assert not loaded.unexpected_keys
    assert set(loaded.missing_keys)=={'supports','blocks.0.graph_conv.identity','blocks.1.graph_conv.identity',
        'blocks.0.temporal_attention.mask','blocks.1.temporal_attention.mask'}
    source = torch.randn(2,60,3,1,dtype=torch.float64,requires_grad=True)
    reference_source = source.detach().clone().requires_grad_()
    # Common data is trend/period/closeness with nearest chunks first; the
    # original ASTGCN data supplied the reverse chunk order to DGCN.
    reference_input = reference_source.reshape(2,5,12,3,1).flip(1).reshape(2,60,3,1)
    torch.manual_seed(79)
    expected = old(dict(X=reference_input))
    torch.manual_seed(79)
    actual = current(dict(X=source))
    torch.testing.assert_close(actual,expected,rtol=5e-9,atol=2e-10)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(source.grad,reference_source.grad,rtol=2e-7,atol=2e-10)
    reference = {native_key(key):value for key,value in old.named_parameters()
        if '.SATT_3.bn.' not in key and '.SATT_2.bn.' not in key}
    assert reference.keys()==dict(current.named_parameters()).keys()
    for key,value in current.named_parameters():
        assert value.grad is not None and reference[key].grad is not None
        torch.testing.assert_close(value.grad,reference[key].grad,rtol=3e-7,atol=3e-10)
    old_state = {native_key(key):value for key,value in old.state_dict().items()}
    for key,value in current.named_buffers():
        if 'running_' in key or 'num_batches_tracked' in key:
            torch.testing.assert_close(value,old_state[key],rtol=1e-12,atol=1e-12)


def test_coarse_attention_projection_observes_whole_chunks_without_interleaving():
    module = SpatialAttention(4,2,chunk=3).double()
    source = torch.arange(2*4*2*12,dtype=torch.float64).reshape(2,4,2,12)
    captured = []
    module.conv1.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
    actual = module(source)
    assert actual.shape==(2,4,2,2)
    packed = captured[0]
    for feature in range(4):
        for position in range(3):
            for chunk in range(4):
                torch.testing.assert_close(packed[:,feature*3+position,:,chunk],source[:,feature,:,chunk*3+position],rtol=0,atol=0)


@pytest.mark.parametrize('order,kernel',[(1,1),(2,3),(3,5)])
def test_dynamic_chebyshev_terms_preserve_feature_order_and_general_temporal_kernel(order,kernel):
    torch.manual_seed(3)
    module = DynamicChebyshev(2,4,3,order,kernel).double()
    source = torch.randn(2,2,3,5,dtype=torch.float64)
    adjacency = torch.randn(2,3,3,dtype=torch.float64)*.1
    terms = [torch.eye(3,dtype=torch.float64).expand(2,-1,-1)]
    if order>1:
        terms.append(adjacency)
    for _ in range(2,order):
        terms.append(2*adjacency@terms[-1]-terms[-2])
    channels = []
    for feature in range(2):
        for term in terms:
            channels.append(torch.stack([sum(term[:,node,neighbor,None]*source[:,feature,neighbor]
                for neighbor in range(3)) for node in range(3)],1))
    expected = torch.nn.functional.conv2d(torch.stack(channels,1),module.conv1.weight,module.conv1.bias,padding=(0,kernel//2))
    torch.testing.assert_close(module(source,adjacency),expected,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('nodes',(1,2,3))
def test_unnormalized_scaled_laplacian_and_owned_buffers_handle_empty_and_directed_graphs(nodes):
    empty = np.zeros((nodes,nodes),dtype=np.float64)
    torch.testing.assert_close(scaled_laplacian(empty),-torch.eye(nodes),rtol=0,atol=0)
    if nodes==2:
        graph = np.array([[0,2],[3,0]],dtype=np.float64)
        expected = 2*np.array([[2,-2],[-3,3]])/5-np.eye(2)
    elif nodes==3:
        graph = np.roll(np.eye(3),1,axis=1)*2
        expected = 2*(2*np.eye(3)-graph)/3-np.eye(3)
    else:
        graph,expected = empty,-np.eye(1)
    original = graph.copy()
    np.testing.assert_allclose(scaled_laplacian(graph).numpy(),expected,rtol=1e-6,atol=1e-7)
    np.testing.assert_array_equal(graph,original)


def test_row_normalization_preserves_original_formula_and_guards_cancellation():
    base = torch.tensor([[-1.,-.3],[.2,-1.]],dtype=torch.float64)
    residual = torch.full((2,2),.001,dtype=torch.float64,requires_grad=True)
    expected = torch.diag((base+residual).sum(-1).add(1e-4).reciprocal())@(base+residual)
    torch.testing.assert_close(graph_support(base,residual,1e-4),expected,rtol=1e-15,atol=1e-15)
    cancellation = (-base-torch.eye(2,dtype=torch.float64)*1e-4).requires_grad_()
    value = graph_support(base,cancellation,1e-4)
    value.square().mean().backward()
    assert torch.isfinite(value).all() and torch.isfinite(cancellation.grad).all()


@pytest.mark.parametrize('counts,horizon,nodes,features,outputs,channels,order,kernel,heads',[
    ((1,0,0),1,1,1,1,1,1,1,4),((0,1,0),2,1,2,2,4,2,3,4),
    ((0,0,2),3,3,2,1,6,3,5,2),((1,1,1),2,3,2,2,4,2,1,1),
    ((3,2,1),1,3,1,2,8,3,3,4),((2,1,2),3,1,2,2,4,2,5,2)])
def test_general_cpt_chunks_horizons_features_single_nodes_and_all_parameter_gradients(
        counts,horizon,nodes,features,outputs,channels,order,kernel,heads):
    meta = metadata(counts,horizon,nodes,features,outputs)
    model = DGCN(metadata=meta,input_window=horizon,output_window=horizon,c_out=channels,K=order,Kt=kernel,
        attention_heads=heads,dropout=0).double().eval()
    source = torch.randn(1,sum(counts)*horizon,nodes,features,dtype=torch.float64,requires_grad=True)
    actual = model(dict(X=source))
    assert actual.shape==(1,horizon,nodes,outputs) and torch.isfinite(actual).all()
    actual.square().mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())
    assert torch.isfinite(source.grad).all()
    if order==1:
        assert model.graph_residual is None and all(block.graph_recurrence is None for block in model.blocks)
        assert not any('attention.conv' in key for key,_ in model.named_parameters() if 'temporal_attention' not in key)


def test_eval_is_label_free_sample_independent_and_preserves_graph_input_and_state():
    model = DGCN(metadata=metadata(horizon=2),input_window=2,output_window=2,c_out=4,K=2,dropout=.4).double().eval()
    source = torch.randn(2,10,3,1,dtype=torch.float64,requires_grad=True)
    original = source.detach().clone()
    state = {key:value.clone() for key,value in model.state_dict().items()}
    prediction = model(dict(X=source))
    torch.testing.assert_close(prediction[:1],model(dict(X=source[:1])),rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(prediction,model(dict(X=source,y=torch.full_like(prediction,float('nan')),
        future_features=source.new_full((2,2,3,4),float('nan')))),rtol=0,atol=0)
    prediction[0].sum().backward()
    torch.testing.assert_close(source.grad[1],torch.zeros_like(source.grad[1]),rtol=0,atol=0)
    torch.testing.assert_close(source,original,rtol=0,atol=0)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,state[key],rtol=0,atol=0)
    assert all(not any(hasattr(part,key) for key in ('device','_device','_scaler','_logger')) for part in model.modules())


def test_full_default_protocol_reads_five_twelve_step_chunks_and_forecasts_without_labels(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=8000,nodes=2)
    cfg = dict(model=dict(name='DGCN'),data=dict(dataset='Tiny',data_dir=str(tmp_path/'raw'),cache_dir=str(tmp_path/'cache')),
        runtime=dict(device='cpu',output_dir=str(tmp_path/'runs'),exp_id='default'))
    context = make_context(cfg)
    try:
        assert context.config['data']['output_window']==12 and context.bundle.metadata.input_layout['length']==60
        assert context.bundle.split_summary['train']==768  # 757 real windows, padded to 64.
        batch = {key:value[:1] for key,value in next(iter(context.bundle.test)).items()}
        assert batch['X'].shape==(1,60,2,1)
        assert context.model.eval()(model_inputs(batch,1)).shape==(1,12,2,1)
    finally:
        context.tracker.close()


def test_common_cpt_reordering_train_statistics_cache_and_future_target_isolation(tmp_path):
    folder = periodic_fixture(tmp_path/'raw',name='Tiny',steps=604,nodes=3,features=2,outputs=1)
    cfg = configuration(tmp_path,'data')
    contexts = [make_context(cfg),make_context(cfg)]
    try:
        first,second = contexts
        assert first.bundle.split_summary['cache']['status']=='miss' and second.bundle.split_summary['cache']['status']=='hit'
        dataset = first.bundle.val.dataset
        sample = dataset[0]
        captured = []
        first.model.input_normalization.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
        first.model.eval().ordered_history(sample['X'][None])
        expected = sample['X'].flip(0).permute(2,1,0)[None]
        torch.testing.assert_close(torch.cat(captured,-1),expected,rtol=0,atol=0)
        train,val,test = (getattr(first.bundle,name).dataset for name in ('train','val','test'))
        assert train.target_starts.max()<val.target_starts.min() and val.target_starts.max()<test.target_starts.min()
        np.testing.assert_allclose(first.bundle.scaler.mean,train.targets[:362].mean(),rtol=1e-6)
        path = folder/'Tiny.dyna'
        frame = pd.read_csv(path)
        frame.loc[(frame['dyna_id']%604)>=int(val.target_starts[0]),'value_0'] += 1000
        frame.to_csv(path,index=False)
        changed = make_context(cfg)
        try:
            after = changed.bundle.val.dataset[0]
            torch.testing.assert_close(after['X'],sample['X'],rtol=0,atol=0)
            assert not torch.equal(after['y_raw'],sample['y_raw'])
            assert changed.bundle.split_summary['cache']['status']=='miss'
        finally:
            changed.tracker.close()
    finally:
        for context in contexts:
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


def test_train_evaluate_dropout_bn_adam_exponential_rng_exact_resume(tmp_path):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=604,nodes=3,features=2,outputs=2)
    trained = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    partial = make_context(configuration(tmp_path,'partial'))
    initial = partial.model.graph_residual.detach().clone()
    train_loop(partial,until_epoch=1)
    assert not torch.equal(initial,partial.model.graph_residual)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(1,3,2)


@pytest.mark.parametrize('single_chunk',(False,True))
def test_real_single_node_tail_cpu_amp_and_singleton_normalization(tmp_path,single_chunk):
    atomic_fixture(tmp_path/'raw',steps=44 if single_chunk else 604,nodes=1,interval_minutes=60)
    cfg = configuration(tmp_path,'amp')
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    if single_chunk:
        cfg['data']['sampling'] = dict(name='cpt',closeness=1,period=0,trend=0)
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


@pytest.mark.parametrize('params',[dict(c_out=3),dict(K=0),dict(Kt=2),dict(attention_heads=0),
    dict(dropout=-.1),dict(normalization_epsilon=0),dict(normalization_epsilon=float('nan'))])
def test_invalid_model_parameters_fail_early(params):
    with pytest.raises(ValueError,match='DGCN'):
        DGCN(metadata=metadata(),input_window=12,output_window=12,**params)


def test_horizon_protocol_loss_and_shape_guards():
    cfg = load_config(config=dict(model=dict(name='DGCN'),data=dict(dataset='Tiny')))
    assert cfg['data']['output_window']==12 and cfg['training']['loss']==dict(name='masked_mse',mask_value=None)
    assert cfg['training']['scheduler']['interval']=='epoch' and cfg['training']['early_stopping']['patience']==0
    with pytest.raises(ValueError,match='square adjacency'):
        scaled_laplacian(np.array([[float('nan')]]))
    plain = DatasetMetadata('bad',1,1,1,adjacency=np.zeros((1,1)),input_layout=dict(kind='window'))
    with pytest.raises(ValueError,match='CPT sampling'):
        DGCN(metadata=plain,input_window=1,output_window=1)
    model = DGCN(metadata=metadata(horizon=1),input_window=1,output_window=1,c_out=4,K=2)
    with pytest.raises(ValueError,match='X must'):
        model(dict(X=torch.zeros(1,4,3,1)))
