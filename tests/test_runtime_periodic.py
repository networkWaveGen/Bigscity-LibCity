"""Correct old periodic graph math explicitly, without certifying broken paths."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from libcity.config.runtime import load_config
from libcity.data.distance_bands import partition_distances
from libcity.data.periodic_offsets import periodic_offsets
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import periodic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,name,experiment):
    return dict(data=dict(dataset='PeriodicTiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            pad_training=False,input_window=3),
        model=dict(name=name,params=dict(conv_depth=2,conv_height=3,hidden_size=4) if name=='CONVGCN' else {}),
        training=dict(epochs=3,batch_size=16,scheduler=dict(name='step',interval='epoch',step_size=1,gamma=.8)),
        evaluation=dict(batch_size=17),runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=experiment))


@pytest.mark.parametrize('features,horizon',[(1,1),(2,1),(1,2),(2,2)])
def test_convgcn_original_graph_projection_corrected_conv_channel_axes_and_all_gradients(tmp_path,features,horizon):
    periodic_fixture(tmp_path/'raw',features=features,outputs=features)
    config = configuration(tmp_path,'CONVGCN','reference')
    config['data']['output_window'] = horizon
    context = make_context(config)
    try:
        model = context.model.eval()
        meta = context.bundle.metadata
        old_type = importlib.import_module('libcity.model.traffic_flow_prediction.CONVGCN').CONVGCN
        old = old_type(dict(context.config['model']['params'],time_lag=4,output_window=horizon,device=torch.device('cpu')),
            dict(num_nodes=4,feature_dim=features,output_dim=features,adj_mx=meta.adjacency.copy(),scaler=NoneScaler())).eval()
        loaded = model.load_state_dict(old.state_dict(),strict=False)
        assert loaded.missing_keys==['adj_mx'] and not loaded.unexpected_keys
        batch = {key:value[:2] for key,value in next(iter(context.bundle.train)).items()}
        projected = old.gc(batch['X'],old.adj_mx)
        # The legacy reshape mixed graph-time and node axes. Node is the learned
        # Conv3d channel; use explicit ordering for the reference correction.
        packed = projected.permute(0,2,1,3).reshape(-1,4,2,3,features)
        expected = old.fc(old.relu(old.Conv(packed)).flatten(1)).reshape(-1,horizon,4,features)
        actual = model(model_inputs(batch,features))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        assert not torch.allclose(actual,old(batch))
        actual.square().mean().backward()
        expected.square().mean().backward()
        for key,value in model.named_parameters():
            assert value.grad is not None
            torch.testing.assert_close(value.grad,dict(old.named_parameters())[key].grad,rtol=1e-4,atol=1e-6)
        np.testing.assert_array_equal(meta.adjacency.diagonal(),np.zeros(4))
        torch.testing.assert_close(model.adj_mx.diag(),torch.ones(4))
    finally:
        context.tracker.close()


@pytest.mark.parametrize('features',[1,2])
def test_multistgcnet_corrected_recent_daily_branch_and_finite_band_reference_gradients(tmp_path,features):
    periodic_fixture(tmp_path/'raw',features=features,outputs=features)
    context = make_context(configuration(tmp_path,'MultiSTGCnet','reference'))
    try:
        model = context.model.eval()
        meta = context.bundle.metadata
        old_type = importlib.import_module('libcity.model.traffic_flow_prediction.MultiSTGCnet').MultiSTGCnet
        old = old_type(dict(device=torch.device('cpu')),dict(num_nodes=4,feature_dim=features,output_dim=features,
            len_closeness=3,len_period=1,len_trend=1,adj_mx=np.ones((4,4)),scaler=NoneScaler())).eval()
        loaded = model.load_state_dict(old.state_dict(),strict=False)
        assert len(loaded.missing_keys)==3 and all(key.endswith('.S') for key in loaded.missing_keys)
        assert not loaded.unexpected_keys
        expected_bands = np.zeros((3,4,4),dtype=np.float32)
        for origin in range(4):
            for hop in range(1,4):
                expected_bands[hop-1,origin,(origin+hop)%4] = 1
        np.testing.assert_array_equal(meta.input_layout['distance_bands'],expected_bands)
        for name,matrix in zip(('near_block','middle_block','distant_block'),expected_bands):
            getattr(old.spatial_component,name).S = torch.tensor(matrix)
        batch = {key:value[:2] for key,value in next(iter(context.bundle.train)).items()}
        source = batch['X']
        temporal = old.temporal_component
        # The old forward called daily_block for both recent and daily data,
        # leaving interval_block untrained. Match corrected branch identities.
        pieces = [temporal.interval_block(source[:,:3]),temporal.daily_block(source[:,3:4]),
                  temporal.weekly_block(source[:,4:])]
        temporal_prediction = torch.relu(temporal.linear(torch.cat(pieces,1)))
        expected = (old.Ws*old.spatial_component(source)+old.Wt*temporal_prediction).reshape(-1,1,4,features)
        actual = model(model_inputs(batch,features))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for key,value in model.named_parameters():
            assert value.grad is not None
            torch.testing.assert_close(value.grad,dict(old.named_parameters())[key].grad,rtol=1e-4,atol=1e-6)
        assert model.temporal_component.interval_block.linear.weight.grad is not None
    finally:
        context.tracker.close()


@pytest.mark.parametrize('distances',[
    [[np.inf,0,1,2],[3,np.inf,3,3],[np.inf,np.inf,np.inf,np.inf],[0,0,0,np.inf]],
    [[np.inf]],[[0]],[[np.inf,1],[1,np.inf]],[[0,1,2,3]]*4,
])
def test_distance_bands_cover_every_listed_edge_without_mutating_graph(distances):
    source = np.asarray(distances,dtype=np.float64)
    original = source.copy()
    bands = partition_distances(source)
    np.testing.assert_array_equal(source,original)
    np.testing.assert_array_equal(bands.sum(0),np.isfinite(source).astype(np.float32))
    for row,values in enumerate(source):
        finite = values[np.isfinite(values)]
        if not len(finite):
            assert not bands[:,row].any()
        elif finite.min()==finite.max():
            np.testing.assert_array_equal(bands[0,row],np.isfinite(values))
        else:
            minimum,maximum = float(finite.min()),float(finite.max())
            for column,value in enumerate(values):
                if not np.isfinite(value):
                    continue
                expected = 0 if value<minimum+(maximum-minimum)/3 else 1 if value<minimum+2*(maximum-minimum)/3 else 2
                assert bands[expected,row,column]==1


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


@pytest.mark.parametrize('name',['CONVGCN','MultiSTGCnet'])
def test_full_train_evaluate_and_exact_optimizer_schedule_rng_resume(tmp_path,name):
    periodic_fixture(tmp_path/'raw')
    trained = run_experiment('train',config=configuration(tmp_path,name,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,name,'evaluated'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    info = json.loads(trained.artifacts['runtime_info'].read_text())
    assert info['benchmark_protocol']==('convgcn_windows' if name=='CONVGCN' else 'multi_stgc')
    partial = make_context(configuration(tmp_path,name,'partial'))
    initial = {key:value.detach().clone() for key,value in partial.model.named_parameters()}
    train_loop(partial,until_epoch=1)
    assert any(not torch.equal(initial[key],value) for key,value in partial.model.named_parameters())
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,name,'resumed'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape==(80,1,4,2)


def test_seasonal_windows_preserve_old_inputs_fix_second_target_and_keep_last_valid_target(tmp_path):
    from libcity.data.dataset.dataset_subclass.convgcn_dataset import CONVGCNDataset
    old = object.__new__(CONVGCNDataset)
    old.time_lag,old.output_window,old.TG_in_one_week,old.TG_in_one_day,old.num_nodes = 3,2,7,4,2
    source = np.arange(20*2*2,dtype=np.float64).reshape(20,2,2)
    old_x,old_y = old._generate_input_data(source)
    offsets,layout = periodic_offsets(dict(name='seasonal_windows',period_steps=4,trend_steps=7),3600,2)
    assert offsets.tolist()==[-9,-8,-6,-5,-2,-1]
    assert layout['segment_order']==['trend','period','closeness']
    np.testing.assert_array_equal(old_x[0],source[9+offsets])
    np.testing.assert_array_equal(old_y[0],source[9:11])
    assert not np.array_equal(old_y[...,0],old_y[...,1])
    periodic_fixture(tmp_path/'raw',steps=40,nodes=2)
    cfg = configuration(tmp_path,'CONVGCN','targets')
    cfg['data'].update(input_window=2,output_window=2,sampling=dict(period_steps=4,trend_steps=7))
    context = make_context(cfg)
    try:
        dataset = context.bundle.test.dataset
        assert dataset.target_starts[-1]==38
        np.testing.assert_array_equal(dataset[-1]['y_raw'].numpy(),dataset.targets[38:40].astype(np.float32))
        assert not torch.equal(dataset[-1]['y_raw'][...,0],dataset[-1]['y_raw'][...,1])
    finally:
        context.tracker.close()


@pytest.mark.parametrize('counts',[(2,0,0),(0,1,0),(0,0,1),(2,1,1)])
def test_multistgcnet_active_branches_multiple_horizons_and_target_dimensions(tmp_path,counts):
    periodic_fixture(tmp_path/'raw',features=3,outputs=2)
    cfg = configuration(tmp_path,'MultiSTGCnet','branches')
    cfg['data'].update(output_window=3,sampling=dict(zip(('closeness','period','trend'),counts)))
    context = make_context(cfg)
    try:
        batch = {key:value[:1] for key,value in next(iter(context.bundle.test)).items()}
        model = context.model.double().eval()
        predicted = model(dict(X=batch['X'].double()))
        assert predicted.shape==(1,3,4,2)
        predicted.sum().backward()
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())
        assert hasattr(model,'spatial_component')==bool(counts[0])
        assert hasattr(model.temporal_component,'interval_block')==bool(counts[0])
        assert hasattr(model.temporal_component,'daily_block')==bool(counts[1])
        assert hasattr(model.temporal_component,'weekly_block')==bool(counts[2])
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name',['CONVGCN','MultiSTGCnet'])
@pytest.mark.parametrize('nodes',[1,4])
def test_no_label_inputs_float64_batch_one_and_stateless_graphs(tmp_path,name,nodes):
    periodic_fixture(tmp_path/'raw',nodes=nodes,features=3,outputs=2)
    config = configuration(tmp_path,name,'dtype')
    config['data']['output_window'] = 2
    context = make_context(config)
    try:
        batch = {key:value[:1] for key,value in next(iter(context.bundle.test)).items()}
        model = context.model.eval().double()
        source = batch['X'].double()
        original = source.clone()
        state = {key:value.clone() for key,value in model.state_dict().items()}
        rng = torch.get_rng_state()
        prediction = model(dict(X=source))
        assert prediction.shape==(1,2,nodes,2) and prediction.dtype==torch.float64
        torch.testing.assert_close(prediction,model(dict(X=source,y=torch.full_like(prediction,float('nan')))),rtol=0,atol=0)
        equal(state,model.state_dict())
        torch.testing.assert_close(source,original,rtol=0,atol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        assert all(not any(hasattr(part,key) for key in ('device','_device','_logger','_scaler')) for part in model.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name',['CONVGCN','MultiSTGCnet'])
def test_actual_cpu_amp_training(tmp_path,name):
    periodic_fixture(tmp_path/'raw')
    cfg = configuration(tmp_path,name,'amp')
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


@pytest.mark.parametrize('mode',['distance_bands','binary'])
def test_distance_graph_direction_symmetry_and_original_source_ownership(tmp_path,mode):
    folder = periodic_fixture(tmp_path/'raw')
    before = pd.read_csv(folder/'PeriodicTiny.rel')
    cfg = configuration(tmp_path,'MultiSTGCnet','symmetry')
    cfg['data']['graph'] = dict(name=mode,symmetric=True,self_loops=True)
    context = make_context(cfg)
    try:
        graph = context.bundle.metadata.input_layout['distance_bands']
        assert not graph.flags.writeable
        np.testing.assert_array_equal(graph,graph.transpose(0,2,1))
        np.testing.assert_array_equal(graph.sum(0),np.ones((4,4)))
        np.testing.assert_array_equal(graph[0].diagonal(),np.ones(4))
        pd.testing.assert_frame_equal(pd.read_csv(folder/'PeriodicTiny.rel'),before)
        if mode=='binary':
            assert context.bundle.split_summary['distance_bands']['column'] is None
    finally:
        context.tracker.close()


@pytest.mark.parametrize('problem',['negative','missing-column','unknown-node'])
def test_invalid_distance_relations_are_rejected(tmp_path,problem):
    folder = periodic_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'MultiSTGCnet','invalid')
    edges = pd.read_csv(folder/'PeriodicTiny.rel')
    if problem=='negative':
        edges.loc[0,'distance'] = -1
    elif problem=='unknown-node':
        edges.loc[0,'destination_id'] = 10000
    else:
        config['data']['graph'] = dict(distance_column='absent')
    edges.to_csv(folder/'PeriodicTiny.rel',index=False)
    with pytest.raises(ValueError,match='distance|Distance'):
        make_context(config)


@pytest.mark.parametrize('name,sampling',[('CONVGCN','window'),('MultiSTGCnet','cpt'),('GRU','frame_cpt'),('GRU','seasonal_windows')])
def test_periodic_protocol_and_model_must_match(name,sampling):
    with pytest.raises(ValueError,match='seasonal_windows|frame_cpt'):
        load_config(config=dict(data=dict(dataset='unused',sampling=dict(name=sampling)),model=dict(name=name)))


def test_distance_column_switch_rebuilds_graph_and_source_changes_reject_resume(tmp_path):
    folder = periodic_fixture(tmp_path/'raw')
    edges = pd.read_csv(folder/'PeriodicTiny.rel')
    edges['other'] = 8-edges['distance']
    edges.to_csv(folder/'PeriodicTiny.rel',index=False)
    cfg = configuration(tmp_path,'MultiSTGCnet','first')
    first = make_context(cfg)
    train_loop(first,until_epoch=1)
    first.tracker.close()
    cfg2 = configuration(tmp_path,'MultiSTGCnet','column')
    cfg2['data']['graph'] = dict(distance_column='other')
    second = make_context(cfg2)
    try:
        assert second.bundle.split_summary['cache']['status']=='miss'
        assert second.bundle.split_summary['distance_bands']['column']=='other'
        assert not np.array_equal(first.bundle.metadata.input_layout['distance_bands'],second.bundle.metadata.input_layout['distance_bands'])
    finally:
        second.tracker.close()
    edges['distance'] = edges['distance'].astype(float)
    edges.loc[0,'distance'] += .125
    edges.to_csv(folder/'PeriodicTiny.rel',index=False)
    with pytest.raises(ValueError,match='metadata/normalization|atomic source files changed'):
        run_experiment('train',config=configuration(tmp_path,'MultiSTGCnet','rejected'),resume=first.directory/'checkpoints/last.pt')
