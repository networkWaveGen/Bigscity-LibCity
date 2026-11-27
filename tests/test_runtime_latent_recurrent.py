"""Factor interpolation, relation axes and explicit graph-filtered recurrence."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.data.bundle import DatasetMetadata
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_flow_prediction.STNN import STNN as ReferenceSTNN
from libcity.model.traffic_flow_prediction.runtime_stnn import STNN
from libcity.model.traffic_speed_prediction.TGCLSTM import TGCLSTM as ReferenceTGCLSTM
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,tgclstm_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,name,experiment,mode='refine'):
    config = tiny_config(root,experiment,epochs=2,batch_size=8,shuffle=True)
    config['model'] = dict(name=name,params=(dict(mode=mode,nhid=4,nlayers=2,dropout_f=.1,dropout_d=.1)
        if name=='STNN' else dict(hops=2)))
    config['training']['scheduler'] = dict(name='step',interval='epoch',step_size=1,gamma=.8)
    return config


def relation_reference(model):
    if model.mode is None:
        return model.relations
    weights = model.rel_weights.clamp(0,1)
    inter = torch.zeros_like(model.relations[:,1:],dtype=weights.dtype)
    if model.mode=='discover':
        inter = weights
    else:
        cursor = 0
        rows = []
        for node in range(model.nodes):
            values = []
            for neighbor in range(model.nodes):
                if model.relations[node,1,neighbor]:
                    values.append(weights[cursor]);cursor += 1
                else:
                    values.append(weights.new_zeros(()))
            rows.append(torch.stack(values)[None])
        inter = torch.stack(rows)
    return torch.cat((model.relations[:,:1].to(weights.dtype),inter),dim=1)


def factor_reference(model,source):
    coordinates = model.ffunc(source.flatten(1)).reshape(source.shape[0],model.window,model.nodes,2).sigmoid()
    if model.lookup=='hard':
        time = (coordinates[...,0]*(model.window-1)).ceil().long()
        node = (coordinates[...,1]*(model.nodes-1)).ceil().long()
        return model.factors[time,node]
    # Independent PyTorch grid sampler: x is node, y is time, corners aligned.
    grid = coordinates.flip(-1)*2-1
    bank = model.factors.permute(2,0,1)[None].expand(source.shape[0],-1,-1,-1)
    return F.grid_sample(bank,grid,mode='bilinear',align_corners=True).permute(0,2,3,1)


def dynamic_reference(model,source):
    latent = factor_reference(model,source)
    relations = relation_reference(model)
    for _ in range(model.horizon):
        batches = []
        for batch in range(len(source)):
            times = []
            for time in range(model.window):
                nodes = []
                for node in range(model.nodes):
                    context = relations[node]@latent[batch,time]
                    nodes.append(torch.tanh(model.dynamic(context.flatten())))
                times.append(torch.stack(nodes))
            batches.append(torch.stack(times))
        latent = torch.stack(batches)
    predictions = [model.decoder(latent[:,:,node].flatten(1)).reshape(len(source),model.horizon,model.outputs)
                   for node in range(model.nodes)]
    return torch.stack(predictions,dim=2)


@pytest.mark.parametrize('mode',[None,'refine','discover'])
@pytest.mark.parametrize('lookup',['hard','linear'])
def test_stnn_explicit_node_time_reference_and_gradients(tmp_path,mode,lookup):
    atomic_fixture(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,'STNN','reference',mode)
    config['model']['params'].update(lookup=lookup,dropout_f=0.,dropout_d=0.)
    context = make_context(config)
    try:
        model = context.model.eval()
        reference = copy.deepcopy(model)
        batch = next(iter(context.bundle.train))
        inputs = model_inputs(batch,1)
        wanted = dynamic_reference(reference,inputs['X'])
        actual = model(inputs)
        torch.testing.assert_close(actual,wanted,rtol=1e-5,atol=1e-6)
        actual.square().mean().backward()
        wanted.square().mean().backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-4,atol=1e-7)
        if lookup=='linear':
            assert model.ffunc.weight.grad is not None and model.ffunc.weight.grad.abs().sum()>0
        else:
            assert model.ffunc.weight.grad is None
    finally:
        context.tracker.close()


def test_stnn_hard_lookup_matches_old_12_by_41_factor_selection():
    graph = np.roll(np.eye(41,dtype=np.float32),1,axis=1)
    metadata = DatasetMetadata('OriginalShape',41,1,1,graph,input_layout=dict(kind='window'))
    current = STNN(metadata=metadata,input_window=12,output_window=2,dropout_f=0.,dropout_d=0.,lookup='hard')
    reference = ReferenceSTNN(dict(input_window=12,output_window=2,dropout_f=0.,dropout_d=0.),
        dict(num_nodes=41,feature_dim=1,output_dim=1,adj_mx=graph.copy(),scaler=NoneScaler()))
    current.load_state_dict(reference.state_dict())
    source = torch.rand(2,12,41,1)
    coordinates = reference.sigmo(reference.ffunc(source.flatten(1)).reshape(2,12,41,2))
    expected = reference.factors[(coordinates[...,0]*11).ceil().long(),(coordinates[...,1]*40).ceil().long()]
    torch.testing.assert_close(current.encode(source),expected,rtol=0,atol=0)


@pytest.mark.parametrize('with_reachability',[False,True])
@pytest.mark.parametrize('clamp',[False,True])
def test_tgclstm_original_recurrence_predictions_objective_and_gradients(tmp_path,with_reachability,clamp):
    tgclstm_fixture(tmp_path/'raw')
    config = configuration(tmp_path,'TGCLSTM','reference')
    config['model']['params']['clamp_adjacency'] = clamp
    if not with_reachability:
        config['data']['graph'] = dict(reachability_column=None)
    context = make_context(config)
    try:
        metadata = context.bundle.metadata
        graph = metadata.adjacency.copy()
        features = dict(num_nodes=3,feature_dim=1,output_dim=1,adj_mx=graph.copy(),scaler=NoneScaler())
        if with_reachability:
            features['FFR'] = [metadata.input_layout['reachability'].copy()]
        reference = ReferenceTGCLSTM(dict(K_hop_numbers=2,Clamp_A=clamp,back_length=0,
            dataset_class='TGCLSTMDataset' if with_reachability else 'TrafficSpeedDataset'),features).eval()
        current = context.model.eval()
        missing = current.load_state_dict(reference.state_dict(),strict=False)
        assert not missing.unexpected_keys
        assert all(key in ('supports','cell_filter') or key.endswith('.filter_square_matrix') for key in missing.missing_keys)
        batch = next(iter(context.bundle.train))
        expected = reference.predict(batch)
        actual = current(model_inputs(batch,1))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        expected_loss = reference.calculate_loss(batch)
        actual_loss = context.loss_fn(actual,batch['y'],batch['y_raw']).loss
        torch.testing.assert_close(expected_loss,actual_loss,rtol=1e-5,atol=1e-6)
        expected_loss.backward()
        actual_loss.backward()
        for name,parameter in current.named_parameters():
            torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-4,atol=1e-6)
        np.testing.assert_array_equal(metadata.adjacency,graph)
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in current.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name,mode',[('STNN',None),('STNN','refine'),('STNN','discover'),('TGCLSTM',None)])
def test_full_lifecycle_and_exact_dropout_resume(tmp_path,name,mode):
    (tgclstm_fixture if name=='TGCLSTM' else atomic_fixture)(tmp_path/'raw',steps=60)
    config = lambda experiment: configuration(tmp_path,name,experiment,mode)
    full = run_experiment('train',config=config('full'))
    evaluated = run_experiment('evaluate',config=config('evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(config('partial'))
    initial_mapping = partial.model.ffunc.weight.detach().clone() if name=='STNN' else None
    train_loop(partial,until_epoch=1)
    if initial_mapping is not None:
        assert not torch.equal(partial.model.ffunc.weight,initial_mapping)
    partial.tracker.close()
    resumed = run_experiment('train',config=config('resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert actual['train_state']==expected['train_state']
    assert actual['scheduler']==expected['scheduler']
    for key,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)


@pytest.mark.parametrize('name',['STNN','TGCLSTM'])
def test_multitarget_batch_one_float64_and_input_isolation(tmp_path,name):
    folder = (tgclstm_fixture if name=='TGCLSTM' else atomic_fixture)(tmp_path/'raw',steps=60)
    frame = pd.read_csv(folder/'Tiny.dyna')
    frame['flow'],frame['observation'] = frame['speed']*.5+2,frame['speed']*.1
    frame.to_csv(folder/'Tiny.dyna',index=False)
    path = folder/'config.json';metadata = json.loads(path.read_text())
    metadata.update(data_col=['speed','flow','observation'],output_dim=2)
    path.write_text(json.dumps(metadata))
    config = configuration(tmp_path,name,'dtype')
    config['data']['features'] = dict(time_in_day=True)
    context = make_context(config)
    try:
        model = context.model.eval().double()
        batch = next(iter(context.bundle.test))
        inputs = {key:value[:1].double() for key,value in model_inputs(batch,2).items()}
        original = {key:value.clone() for key,value in inputs.items()}
        before = torch.get_rng_state()
        prediction = model(inputs)
        assert prediction.shape==(1,2,3,2) and prediction.dtype==torch.float64
        assert torch.isfinite(prediction).all()
        torch.testing.assert_close(prediction,model({**inputs,'y':torch.full_like(prediction,float('nan'))}),rtol=0,atol=0)
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
        for key,value in original.items():
            torch.testing.assert_close(value,inputs[key],rtol=0,atol=0)
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger')) for module in model.modules())
        if name=='TGCLSTM':
            with pytest.raises(ValueError,match='future_features'):
                model({'X':inputs['X']})
    finally:
        context.tracker.close()


@pytest.mark.parametrize('name',['STNN','TGCLSTM'])
def test_actual_cpu_amp_training(tmp_path,name):
    (tgclstm_fixture if name=='TGCLSTM' else atomic_fixture)(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,name,'amp')
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0


def test_reachability_uses_geo_order_missing_edges_zero_and_cache_invalidation(tmp_path):
    folder = tgclstm_fixture(tmp_path/'raw')
    for filename,column in (('Tiny.geo','geo_id'),('Tiny.rel','origin_id'),('Tiny.rel','destination_id'),('Tiny.dyna','entity_id')):
        frame = pd.read_csv(folder/filename)
        frame[column] = frame[column].map({0:100,1:30,2:70})
        frame.to_csv(folder/filename,index=False)
    config = configuration(tmp_path,'TGCLSTM','mapping')
    config['data']['cache'] = True
    first = make_context(config)
    second = None
    try:
        reachability = first.bundle.metadata.input_layout['reachability']
        np.testing.assert_allclose(reachability,[[0.,.8/3,0.],[0.,0.,1.6/3],[.8,0.,0.]])
        assert not reachability.flags.writeable
        assert torch.isfinite(first.model.supports).all()
        assert first.bundle.split_summary['cache']['status']=='miss'
        edges = pd.read_csv(folder/'Tiny.rel');edges.loc[0,'FFR_20min'] = .123
        edges.to_csv(folder/'Tiny.rel',index=False)
        config['runtime']['exp_id'] = 'changed'
        second = make_context(config)
        assert second.bundle.split_summary['cache']['status']=='miss'
        assert second.bundle.metadata.input_layout['reachability'][0,1]==pytest.approx(.123)
    finally:
        first.tracker.close()
        if second is not None:
            second.tracker.close()


def test_reachability_missing_invalid_and_explicit_opt_out(tmp_path):
    folder = atomic_fixture(tmp_path/'raw',steps=60)
    config = configuration(tmp_path,'TGCLSTM','missing')
    with pytest.raises(ValueError,match='reachability column'):
        make_context(config)
    config['data']['graph'] = dict(reachability_column=None)
    context = make_context(config)
    context.tracker.close()
    folder = tgclstm_fixture(tmp_path/'raw')
    edges = pd.read_csv(folder/'Tiny.rel');edges.loc[0,'FFR_20min'] = float('nan')
    edges.to_csv(folder/'Tiny.rel',index=False)
    config['data']['graph'] = dict(reachability_column='FFR_20min')
    with pytest.raises(ValueError,match='finite and nonnegative'):
        make_context(config)
