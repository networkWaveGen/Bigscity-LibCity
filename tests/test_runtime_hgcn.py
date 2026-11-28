"""Hierarchical graph aggregation, static partitions and shared training."""
import importlib
import re

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.cluster import SpectralClustering

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.data.hgcn_bundle import canonical_labels,cluster_graph,graph_partition
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_speed_prediction.runtime_hgcn import HGCN,adaptive_support,graph_supports
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,periodic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(nodes=4,features=2,outputs=2,labels=None):
    graph = np.roll(np.eye(nodes,dtype=np.float32),1,axis=1) if nodes>1 else np.zeros((1,1),dtype=np.float32)
    if nodes==4:
        graph = np.array([[0,1,0,0],[.5,0,1,0],[0,0,0,2],[0,0,1,0]],dtype=np.float32)
    labels = np.arange(nodes)%2 if labels is None and nodes>1 else np.zeros(nodes,dtype=np.int64) if labels is None else np.asarray(labels)
    labels = canonical_labels(labels)
    membership = np.eye(int(labels.max())+1,dtype=np.float32)[labels]
    hierarchy = dict(membership=membership,cluster_adjacency=cluster_graph(graph,membership,.1))
    return DatasetMetadata('tiny',nodes,features,outputs,adjacency=graph,input_layout=dict(kind='window',hierarchy=hierarchy))


def configuration(root,name):
    return dict(data=dict(dataset='Tiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=12,output_window=2,pad_training=False,graph=dict(cluster_nodes=2)),
        model=dict(name='HGCN',params=dict(channels=4,skip_channels=8,end_channels=8)),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2),evaluation=dict(batch_size=5),
        runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


def native_key(key):
    for before,after in {'block_cluster1':'cluster_blocks.0','block_cluster2':'cluster_blocks.1',
            'block1':'node_blocks.0','block2':'node_blocks.1','transmit1':'transfers.0','transmit2':'transfers.1',
            'transmit3':'transfers.2','skip_conv1':'skips.0','skip_conv2':'skips.1',
            '.multigcn.mlp.mlp.':'.graph_conv.mlp.','.TAT.':'.temporal_attention.'}.items():
        key = key.replace(before,after)
    key = re.sub(r'^((?:node|cluster)_blocks\.\d+)\.conv1\.',r'\1.residual.',key)
    return re.sub(r'^((?:node|cluster)_blocks\.\d+)\.bn\.',r'\1.normalization.',key)


def corrected_temporal(module):
    def forward(value):
        left = module.conv1(value.permute(0,1,3,2)).squeeze(1)
        right = module.conv2(value.permute(0,2,1,3)).squeeze(1)
        score = module.v@torch.sigmoid(left@module.w@right+module.b)
        return module.bn(score.transpose(1,2)).transpose(1,2).softmax(-1)
    module.forward = forward


PROFILES = [(kind,4,8,8,False) for kind in ('doubletransition','transition','symnadj','normlap','scalap','identity')]
PROFILES += [('doubletransition',4,8,8,True),('doubletransition',32,256,512,False)]


@pytest.mark.parametrize('kind,channels,skip,end,training',PROFILES)
def test_original_hierarchy_prediction_all_effective_gradients_and_bn_with_correct_member_sum(kind,channels,skip,end,training):
    torch.manual_seed(23)
    meta = metadata(labels=[0,0,1,1])
    current = HGCN(metadata=meta,input_window=12,output_window=3,channels=channels,skip_channels=skip,
        end_channels=end,adjtype=kind).double().train(training)
    old_type = importlib.import_module('libcity.model.traffic_speed_prediction.HGCN').HGCN
    old = old_type(dict(device='cpu',cluster_nodes=2,input_window=12,output_window=3,channels=channels,
        skip_channels=skip,end_channels=end,adjtype=kind,dropout=.3),dict(num_nodes=4,feature_dim=2,output_dim=2,
        adj_mx=meta.adjacency.copy(),adj_mx_cluster=meta.input_layout['hierarchy']['cluster_adjacency'].copy(),
        centers_ind_groups=[[0,1],[2,3]],transmit=torch.tensor(meta.input_layout['hierarchy']['membership'].copy()),
        scaler=NoneScaler())).double().train(training)
    old.get_input_cluster = lambda value:torch.stack((value[:,:,:2].sum(2),value[:,:,2:].sum(2)),2)
    # The reference keeps the original graph/transfer/temporal computation but
    # corrects cluster aggregation, singleton axes and non-buffer static dtype.
    old.supports = [value.clone() for value in current.supports]
    old.supports_cluster = [value.clone() for value in current.supports_cluster]
    for transfer in (old.transmit1,old.transmit2,old.transmit3):
        transfer.transmit = current.membership.clone()
    for block in (old.block1,old.block2,old.block_cluster1,old.block_cluster2):
        corrected_temporal(block.TAT)
    with torch.no_grad():
        for name in ('nodevec1','nodevec2','nodevec1_c','nodevec2_c'):
            getattr(old,name).abs_().add_(.1)  # Original valid nonzero-degree adaptive graph branch.
    keep = lambda key:key not in ('h','h_cluster') and not key.startswith('gate')
    loaded = current.load_state_dict({native_key(key):value for key,value in old.state_dict().items() if keep(key)},strict=False)
    assert not loaded.unexpected_keys
    assert all(key in ('membership','supports','supports_cluster','node_identity','cluster_identity') or key.endswith('.mask') for key in loaded.missing_keys)
    source = torch.randn(2,12,4,2,dtype=torch.float64,requires_grad=True)
    reference_source = source.detach().clone().requires_grad_()
    torch.manual_seed(71)
    expected = old(dict(X=reference_source))
    torch.manual_seed(71)
    actual = current(dict(X=source))
    torch.testing.assert_close(actual,expected,rtol=2e-8,atol=2e-10)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(source.grad,reference_source.grad,rtol=2e-7,atol=2e-10)
    reference = {native_key(key):value for key,value in old.named_parameters() if keep(key)}
    assert reference.keys()==dict(current.named_parameters()).keys()
    for key,value in current.named_parameters():
        assert value.grad is not None and reference[key].grad is not None
        torch.testing.assert_close(value.grad,reference[key].grad,rtol=5e-7,atol=3e-10)
    old_state = {native_key(key):value for key,value in old.state_dict().items() if keep(key)}
    for key,value in current.named_buffers():
        if 'running_' in key or 'num_batches_tracked' in key:
            torch.testing.assert_close(value,old_state[key],rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('kind',('doubletransition','transition','symnadj','normlap','scalap','identity'))
def test_static_supports_match_original_directed_formulas_and_own_input(kind):
    source = metadata().adjacency
    original = source.copy()
    old_module = importlib.import_module('libcity.model.traffic_speed_prediction.HGCN')
    expected = old_module.HGCN.cal_adj(None,source,kind)
    actual = graph_supports(source,kind)
    np.testing.assert_allclose(actual.numpy(),np.stack([np.asarray(value) for value in expected]),rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(source,original)
    assert torch.isfinite(graph_supports(np.zeros((1,1)),kind)).all()


def test_every_cluster_member_contributes_once_with_correct_source_gradients():
    model = HGCN(metadata=metadata(labels=[0,0,1,1]),input_window=12,output_window=2,channels=4,skip_channels=8,end_channels=8).double()
    source = torch.arange(2*2*4*12,dtype=torch.float64).reshape(2,2,4,12).requires_grad_()
    actual = model.cluster_inputs(source)
    expected = torch.stack((source[:,:,:2].sum(2),source[:,:,2:].sum(2)),2)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual.sum(2),source.sum(2),rtol=0,atol=0)
    actual[:,:,0].sum().backward()
    torch.testing.assert_close(source.grad[:,:,:2],torch.ones_like(source.grad[:,:,:2]),rtol=0,atol=0)
    assert torch.count_nonzero(source.grad[:,:,2:])==0


def test_zero_degree_adaptive_rows_and_constant_cluster_graph_are_finite():
    left = torch.tensor([[1.,0.],[-1.,0.]],dtype=torch.float64,requires_grad=True)
    right = torch.tensor([[-1.,-2.],[0.,0.]],dtype=torch.float64,requires_grad=True)
    actual = adaptive_support(left,right,torch.eye(2,dtype=torch.float64))
    torch.testing.assert_close(actual,torch.tensor([[1.,0.],[1/3,2/3]],dtype=torch.float64),rtol=0,atol=0)
    actual.square().sum().backward()
    assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()
    for graph in (np.zeros((4,4)),np.ones((4,4))):
        membership = np.eye(2)[[0,0,1,1]]
        np.testing.assert_array_equal(cluster_graph(graph,membership,.1),np.zeros((2,2)))


def test_spectral_partition_is_seeded_static_and_uses_private_numpy_rng():
    graph = metadata(nodes=4).adjacency.copy()
    affinity = (graph+graph.T)/2
    np.fill_diagonal(affinity,0.)
    expected = SpectralClustering(n_clusters=2,affinity='precomputed',assign_labels='discretize',random_state=17).fit_predict(affinity)
    state = np.random.get_state()
    actual = graph_partition(graph,2,17)
    np.testing.assert_array_equal(actual,canonical_labels(expected))
    np.testing.assert_array_equal(actual,graph_partition(graph,2,17))
    after = np.random.get_state()
    assert state[0]==after[0] and state[2:]==after[2:]
    np.testing.assert_array_equal(state[1],after[1])


@pytest.mark.parametrize('nodes,count',[(1,0),(4,0),(4,1),(4,2)])
def test_degenerate_partition_handles_singleton_all_nodes_and_empty_graph(nodes,count):
    labels = graph_partition(np.zeros((nodes,nodes)),count,17)
    actual_count = min(10,nodes) if count==0 else count
    assert labels.shape==(nodes,) and len(np.unique(labels))==actual_count


def test_explicit_partition_preserves_node_order_and_gaussian_sum_formula():
    graph = metadata().adjacency.copy()
    labels = graph_partition(graph,2,17,labels=[9,9,2,2])
    np.testing.assert_array_equal(labels,[0,0,1,1])
    membership = np.eye(2)[labels]
    sums = np.array([[sum(float(graph[i,j]) for i in np.flatnonzero(labels==source)
        for j in np.flatnonzero(labels==target)) for target in range(2)] for source in range(2)])
    expected = np.exp(-np.square(sums/sums.std()))
    expected[expected<.1] = 0
    np.testing.assert_allclose(cluster_graph(graph,membership,.1),expected,rtol=1e-6,atol=1e-8)
    with pytest.raises(ValueError,match='each geo-ordered node'):
        graph_partition(graph,0,17,labels=[0,1])
    with pytest.raises(ValueError,match='does not match'):
        graph_partition(graph,3,17,labels=[0,0,1,1])


@pytest.mark.parametrize('history,nodes,labels,features,outputs,order,kernels',[
    (10,1,[0],1,1,1,(3,2)),(12,4,[0,0,0,0],2,2,2,(3,2)),
    (15,4,[0,1,2,3],3,2,3,(3,2)),(1,1,[0],2,2,2,(1,1)),
    (5,3,[0,0,1],2,1,1,(1,2)),(8,3,[0,1,1],1,3,2,(2,2))])
def test_general_history_feature_target_graph_orders_singletons_and_effective_gradients(history,nodes,labels,features,outputs,order,kernels):
    model = HGCN(metadata=metadata(nodes,features,outputs,labels),input_window=history,output_window=3,
        channels=4,skip_channels=8,end_channels=8,adaptive_dim=3,graph_order=order,temporal_kernels=kernels).double()
    source = torch.randn(1,history,nodes,features,dtype=torch.float64,requires_grad=True)
    actual = model(dict(X=source))
    assert actual.shape==(1,3,nodes,outputs) and torch.isfinite(actual).all()
    actual.square().mean().backward()
    assert torch.isfinite(source.grad).all()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())


def test_eval_is_label_free_batch_independent_and_graph_input_state_are_owned():
    meta = metadata()
    graph = meta.adjacency.copy()
    model = HGCN(metadata=meta,input_window=12,output_window=2,channels=4,skip_channels=8,end_channels=8).double().eval()
    source = torch.randn(2,12,4,2,dtype=torch.float64,requires_grad=True)
    original = source.detach().clone()
    state = {key:value.clone() for key,value in model.state_dict().items()}
    predicted = model(dict(X=source))
    torch.testing.assert_close(predicted[:1],model(dict(X=source[:1])),rtol=1e-9,atol=1e-11)
    torch.testing.assert_close(predicted,model(dict(X=source,y=torch.full_like(predicted,float('nan')),
        future_features=source.new_full((2,2,4,3),float('nan')))),rtol=0,atol=0)
    predicted[0].sum().backward()
    assert torch.count_nonzero(source.grad[1])==0
    torch.testing.assert_close(source,original,rtol=0,atol=0)
    np.testing.assert_array_equal(meta.adjacency,graph)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,state[key],rtol=0,atol=0)
    assert all(not any(hasattr(part,key) for key in ('device','_device','_scaler','_logger')) for part in model.modules())


def test_partition_settings_reuse_raw_cache_and_future_targets_do_not_change_hierarchy(tmp_path):
    folder = periodic_fixture(tmp_path/'raw',name='Tiny',steps=60,nodes=4,features=2,outputs=2)
    cfg = configuration(tmp_path,'data')
    cfg['data']['graph']['cluster_labels'] = [0,0,1,1]
    first = make_context(cfg)
    cfg['data']['graph']['cluster_labels'] = [0,1,0,1]
    second = make_context(cfg)
    try:
        assert first.bundle.split_summary['cache']['status']=='miss' and second.bundle.split_summary['cache']['status']=='hit'
        assert first.bundle.split_summary['cache']['path']==second.bundle.split_summary['cache']['path']
        assert not np.array_equal(first.bundle.metadata.input_layout['hierarchy']['membership'],second.bundle.metadata.input_layout['hierarchy']['membership'])
        hierarchy = second.bundle.metadata.input_layout['hierarchy']
        assert not hierarchy['membership'].flags.writeable and not hierarchy['cluster_adjacency'].flags.writeable
        sample = second.bundle.val.dataset[0]
        start = int(second.bundle.val.dataset.target_starts[0])
        path = folder/'Tiny.dyna'
        frame = pd.read_csv(path)
        frame.loc[(frame['dyna_id']%60)>=start,['value_0','value_1']] += 1000
        frame.to_csv(path,index=False)
        changed = make_context(cfg)
        try:
            np.testing.assert_array_equal(changed.bundle.metadata.input_layout['hierarchy']['membership'],hierarchy['membership'])
            np.testing.assert_array_equal(changed.bundle.metadata.input_layout['hierarchy']['cluster_adjacency'],hierarchy['cluster_adjacency'])
            torch.testing.assert_close(changed.bundle.val.dataset[0]['X'],sample['X'],rtol=0,atol=0)
            assert not torch.equal(changed.bundle.val.dataset[0]['y_raw'],sample['y_raw'])
            assert changed.bundle.scaler.mean==second.bundle.scaler.mean
        finally:
            changed.tracker.close()
    finally:
        first.tracker.close();second.tracker.close()


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


def test_spectral_hierarchy_train_evaluate_dropout_bn_adam_schedule_rng_exact_resume(tmp_path):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=60,nodes=4,features=2,outputs=2)
    trained = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    partial = make_context(configuration(tmp_path,'partial'))
    initial = partial.model.transfers[0].w.detach().clone()
    train_loop(partial,until_epoch=1)
    assert not torch.equal(initial,partial.model.transfers[0].w)
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(2,4,2)


@pytest.mark.parametrize('singleton_time',(False,True))
def test_actual_cpu_amp_single_node_single_region_and_single_time_tail(tmp_path,singleton_time):
    atomic_fixture(tmp_path/'raw',steps=46 if singleton_time else 48,nodes=1)
    cfg = configuration(tmp_path,'amp')
    cfg['data'].update(input_window=1 if singleton_time else 12,output_window=1,graph=dict(cluster_nodes=0))
    if singleton_time:
        cfg['model']['params']['temporal_kernels'] = [1,1]
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


@pytest.mark.parametrize('params',[dict(channels=0),dict(dropout=1),dict(adaptive_dim=0),dict(graph_order=0),
    dict(temporal_kernels=[3]),dict(temporal_kernels=[3,0]),dict(adjtype='bad')])
def test_invalid_model_parameters_fail_early(params):
    with pytest.raises(ValueError,match='HGCN'):
        HGCN(metadata=metadata(),input_window=12,output_window=2,**params)


@pytest.mark.parametrize('options',[dict(cluster_nodes=-1),dict(cluster_nodes=True),dict(cluster_labels=[]),
    dict(cluster_labels=[True]),dict(cluster_epsilon=1.1),dict(cluster_epsilon=float('nan'))])
def test_invalid_hierarchy_configuration_is_rejected_before_data(options):
    with pytest.raises(ValueError,match='graph.cluster'):
        load_config(config=dict(model=dict(name='HGCN'),data=dict(dataset='Tiny',graph=options)))


def test_metadata_partition_input_length_and_nonhierarchical_config_guards():
    with pytest.raises(ValueError,match='input_window is too short'):
        HGCN(metadata=metadata(),input_window=9,output_window=2)
    with pytest.raises(ValueError,match='cluster_nodes'):
        graph_partition(np.zeros((2,2)),3,17)
    with pytest.raises(ValueError,match='Unknown configuration keys'):
        load_config(config=dict(model=dict(name='GRU'),data=dict(dataset='Tiny',graph=dict(cluster_nodes=2))))
    model = HGCN(metadata=metadata(),input_window=12,output_window=2,channels=4,skip_channels=8,end_channels=8)
    with pytest.raises(ValueError,match='X must'):
        model(dict(X=torch.zeros(1,11,4,2)))
