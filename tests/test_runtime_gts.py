"""GTS preserves train-time graph math with an explicit deterministic evaluation."""
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from libcity.config.runtime import load_config
from libcity.data.bundle import DatasetMetadata
from libcity.executor.evaluate import evaluate_loop,model_inputs
from libcity.executor.gts_train import evaluate_gts,graph_step,teacher_probability,train
from libcity.executor.train import TrainState
from libcity.model.traffic_speed_prediction.runtime_gts import GTS,DiffusionCell,sampled_graph
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture,periodic_fixture
from libcity.utils.normalization import NoneScaler
from libcity.utils.runtime_checkpoint import data_state
from runtime_helpers import make_context

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def metadata(nodes=3,features=2,outputs=2,points=25):
    history = np.sin(np.arange(points)[:,None]/3+np.arange(nodes)[None,:]*.7).astype(np.float32)
    graph = np.roll(np.eye(nodes,dtype=np.float32),1,axis=1) if nodes>1 else np.zeros((1,1),dtype=np.float32)
    return DatasetMetadata('tiny',nodes,features,outputs,adjacency=graph,
        input_layout=dict(kind='window',graph_history=dict(source='raw_training_interval',shape=[points,nodes])),
        static_data=dict(graph_history=history))


def configuration(root,name):
    return dict(data=dict(dataset='Tiny',data_dir=str(root/'raw'),cache_dir=str(root/'cache'),
            input_window=4,output_window=2,pad_training=False),
        model=dict(name='GTS',params=dict(rnn_units=4,num_rnn_layers=2,max_diffusion_step=2,kernel_size=3,embedding_dim=8)),
        training=dict(epochs=3,batch_size=5,accumulation_steps=2,
            strategy=dict(graph_regularization_epochs=1,cl_decay_steps=5)),
        evaluation=dict(batch_size=2),runtime=dict(device='cpu',seed=17,output_dir=str(root/'runs'),exp_id=name))


@pytest.mark.parametrize('units,layers,depth,kernel,embedding,features,outputs,steps,horizon,training',[
    (64,1,3,10,100,1,1,12,12,False),(4,2,2,3,6,3,2,4,3,True),
    (3,1,0,1,4,2,2,1,1,False),(4,1,1,2,5,2,1,5,2,False)])
def test_original_full_gumbel_forecast_logits_auxiliary_and_all_gradients_with_controlled_noise(
        monkeypatch,units,layers,depth,kernel,embedding,features,outputs,steps,horizon,training):
    torch.manual_seed(13)
    meta = metadata(features=features,outputs=outputs,points=2*kernel+7)
    current = GTS(metadata=meta,input_window=steps,output_window=horizon,rnn_units=units,num_rnn_layers=layers,
        max_diffusion_step=depth,kernel_size=kernel,embedding_dim=embedding).double().train(training)
    source_module = importlib.import_module('libcity.model.traffic_speed_prediction.GTS')
    old = source_module.GTS(dict(device=torch.device('cpu'),rnn_units=units,num_rnn_layers=layers,
        max_diffusion_step=depth,kernal_size=kernel,embedding_dim=embedding,input_window=steps,
        output_window=horizon,output_dim=outputs,use_curriculum_learning=False),
        dict(num_nodes=3,feature_dim=features,output_dim=outputs,adj_mx=meta.adjacency.copy(),num_batches=1,
            train_data=meta.static_data['graph_history'].copy(),scaler=NoneScaler())).double().train(training)
    old.node_feas = old.node_feas.double()
    old.rel_rec,old.rel_send,old.adj_mx = old.rel_rec.double(),old.rel_send.double(),old.adj_mx.double()
    old._prepare_data_x = lambda value:value.permute(1,0,2,3).reshape(steps,len(value),3*features)
    loaded = current.load_state_dict(old.state_dict(),strict=False)
    assert not loaded.unexpected_keys and set(loaded.missing_keys)=={'node_features','graph_prior','identity','off_diagonal'}
    uniforms = .1+.8*torch.rand((3,3,2),dtype=torch.float64,generator=torch.Generator().manual_seed(99))
    monkeypatch.setattr(source_module,'sample_gumbel',lambda device,shape,eps=0:-torch.log(-torch.log(uniforms.reshape(shape))))
    source = torch.randn(2,steps,3,features,dtype=torch.float64,requires_grad=True)
    reference_source = source.detach().clone().requires_grad_()
    expected,reference_logits = old(dict(X=reference_source,y=None))
    actual,logits = current.forward_with_aux(dict(X=source),uniforms=uniforms)
    torch.testing.assert_close(actual,expected,rtol=3e-9,atol=2e-11)
    torch.testing.assert_close(logits,reference_logits,rtol=3e-10,atol=2e-11)
    expected_objective = expected.square().mean()+.2*F.binary_cross_entropy_with_logits(reference_logits,old.adj_mx)
    actual_objective = actual.square().mean()+.2*F.binary_cross_entropy_with_logits(logits,current.graph_prior)
    expected_objective.backward();actual_objective.backward()
    torch.testing.assert_close(source.grad,reference_source.grad,rtol=2e-7,atol=2e-10)
    reference = dict(old.named_parameters())
    assert reference.keys()==dict(current.named_parameters()).keys()
    for key,value in current.named_parameters():
        assert value.grad is not None and reference[key].grad is not None
        torch.testing.assert_close(value.grad,reference[key].grad,rtol=2e-7,atol=2e-10)
    reference_buffers = dict(old.named_buffers())
    for key,value in current.named_buffers():
        if key in reference_buffers:
            torch.testing.assert_close(value,reference_buffers[key],rtol=1e-11,atol=1e-11)


@pytest.mark.parametrize('depth',(0,1,3))
def test_dynamic_diffusion_cell_original_recurrence_and_graph_gradients(depth):
    torch.manual_seed(7)
    original = importlib.import_module('libcity.model.traffic_speed_prediction.GTS').DCGRUCell(
        2,3,depth,3,'cpu').double()
    current = DiffusionCell(2,3,depth).double()
    current.load_state_dict(original.state_dict())
    source = torch.randn(2,3,2,dtype=torch.float64,requires_grad=True)
    hidden = torch.randn(2,3,3,dtype=torch.float64,requires_grad=True)
    adjacency = torch.rand(3,3,dtype=torch.float64,requires_grad=True)
    a,b,c = (value.detach().clone().requires_grad_() for value in (source,hidden,adjacency))
    expected = original(a.flatten(1),b.flatten(1),c).reshape(2,3,3)
    graph = adjacency+torch.eye(3,dtype=torch.float64)
    actual = current(source,hidden,(graph/graph.sum(1,keepdim=True)).T)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    expected.square().mean().backward();actual.square().mean().backward()
    for value,reference in ((source,a),(hidden,b)):
        torch.testing.assert_close(value.grad,reference.grad,rtol=1e-10,atol=1e-12)
    if depth:
        torch.testing.assert_close(adjacency.grad,c.grad,rtol=1e-10,atol=1e-12)
    else:
        assert adjacency.grad is None and c.grad is None


def test_factored_pair_logits_and_all_cnn_classifier_gradients_match_explicit_pairs():
    torch.manual_seed(17)
    model = GTS(metadata=metadata(nodes=5,points=13),input_window=4,output_window=2,
        rnn_units=4,kernel_size=3,embedding_dim=7).double().eval()
    actual = model.graph_logits()
    values = model.bn1(model.conv1(model.node_features.T[:,None]).relu())
    values = model.bn2(model.conv2(values).relu())
    values = model.bn3(model.fc(values.flatten(1)).relu())
    pairs = torch.stack([torch.cat((values[sender],values[receiver])) for receiver in range(5) for sender in range(5)])
    expected = model.fc_cat(model.fc_out(pairs).relu()).reshape(5,5,2)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    parameters = [value for name,value in model.named_parameters() if not name.startswith(('encoder_model','decoder_model'))]
    actual_gradients = torch.autograd.grad(actual.square().mean(),parameters)
    expected_gradients = torch.autograd.grad(expected.square().mean(),parameters)
    for value,reference in zip(actual_gradients,expected_gradients,strict=True):
        torch.testing.assert_close(value,reference,rtol=1e-10,atol=1e-12)
    assert not hasattr(model,'rel_rec') and not hasattr(model,'rel_send')


def test_straight_through_gumbel_has_hard_values_and_analytic_soft_gradient():
    logits = torch.tensor([[.3,-.2],[-.4,.1]],dtype=torch.float64,requires_grad=True)
    uniforms = torch.tensor([[.25,.7],[.8,.2]],dtype=torch.float64)
    soft = ((logits-torch.log(-torch.log(uniforms)))/.7).softmax(-1)
    actual = sampled_graph(logits,.7,uniforms)
    torch.testing.assert_close(actual,(soft.argmax(-1)==0).double(),rtol=0,atol=0)
    actual.sum().backward()
    expected = torch.stack((soft[:,0]*soft[:,1],-soft[:,0]*soft[:,1]),1)/.7
    torch.testing.assert_close(logits.grad,expected,rtol=1e-13,atol=1e-13)
    extreme = sampled_graph(torch.zeros(2,2),.5,torch.tensor([[0.,1.],[1.,0.]]))
    assert torch.isfinite(extreme).all()


def test_inference_is_deterministic_label_free_batch_independent_and_does_not_consume_rng():
    model = GTS(metadata=metadata(),input_window=4,output_window=3,rnn_units=4,kernel_size=3,embedding_dim=6).double().eval()
    source = torch.randn(2,4,3,2,dtype=torch.float64,requires_grad=True)
    original = source.detach().clone()
    state = {key:value.clone() for key,value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    actual = model(dict(X=source))
    assert isinstance(actual,torch.Tensor) and actual.shape==(2,3,3,2)
    torch.testing.assert_close(actual,model(dict(X=source,y=torch.full_like(actual,float('nan')),
        future_features=source.new_full((2,3,3,2),float('nan')))),rtol=0,atol=0)
    torch.testing.assert_close(actual[:1],model(dict(X=source[:1])),rtol=1e-10,atol=1e-12)
    actual[0].sum().backward()
    assert torch.count_nonzero(source.grad[1])==0
    torch.testing.assert_close(source,original,rtol=0,atol=0)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    assert numpy_rng[0]==np.random.get_state()[0] and numpy_rng[2:]==np.random.get_state()[2:]
    np.testing.assert_array_equal(numpy_rng[1],np.random.get_state()[1])
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,state[key],rtol=0,atol=0)
    assert all(not any(hasattr(part,key) for key in ('device','_device','_scaler','_logger')) for part in model.modules())


def test_teacher_targets_only_affect_explicit_subsequent_decoder_steps():
    model = GTS(metadata=metadata(),input_window=4,output_window=3,rnn_units=4,kernel_size=3,embedding_dim=6).double().eval()
    source = torch.randn(1,4,3,2,dtype=torch.float64)
    graph = model.inference_graph()
    targets = torch.full((1,3,3,2),10.,dtype=torch.float64)
    autonomous = model.forecast(source,graph)
    teacher = model.forecast(source,graph,targets,[True]*3)
    torch.testing.assert_close(autonomous[:,0],teacher[:,0],rtol=0,atol=0)
    assert not torch.allclose(autonomous[:,1:],teacher[:,1:])
    torch.testing.assert_close(autonomous,model.forecast(source,graph,targets*float('nan'),[False]*3),rtol=0,atol=0)
    assert teacher_probability(0,2)==pytest.approx(2/3)
    assert teacher_probability(10**9,2)==0.


def test_graph_history_uses_full_raw_train_prefix_and_is_independent_of_windows_padding_and_batch_size(tmp_path):
    folder = periodic_fixture(tmp_path/'raw',name='Tiny',steps=60,nodes=4,features=2,outputs=1)
    cfg = configuration(tmp_path,'data')
    first = make_context(cfg)
    cfg['data'].update(input_window=8,output_window=3,pad_training=True)
    cfg['training']['batch_size'] = 7
    second = make_context(cfg)
    cfg['data']['graph'] = dict(history_steps=8,history_channel=1)
    third = make_context(cfg)
    try:
        raw = first.bundle.train.dataset.history
        expected = first.bundle.scaler.transform(raw[:36,:,0]).astype(np.float32)
        np.testing.assert_array_equal(first.bundle.metadata.static_data['graph_history'],expected)
        np.testing.assert_array_equal(second.bundle.metadata.static_data['graph_history'],expected)
        assert first.model.fc.weight.shape==second.model.fc.weight.shape
        assert not first.bundle.metadata.static_data['graph_history'].flags.writeable
        np.testing.assert_array_equal(third.bundle.metadata.static_data['graph_history'],raw[28:36,:,1].astype(np.float32))
        assert third.bundle.metadata.input_layout['graph_history']['normalization']=='unscaled_observation'
        assert second.bundle.split_summary['cache']['status']=='hit' and third.bundle.split_summary['cache']['status']=='hit'
        schema = data_state(first.bundle)
        assert schema['static_data']==dict(graph_history=dict(shape=[36,4],dtype='float32'))
        assert 'values' not in schema['input_layout']['graph_history']
        source = first.bundle.val.dataset[0]
        path = folder/'Tiny.dyna'
        frame = pd.read_csv(path)
        frame.loc[(frame['dyna_id']%60)>=36,['value_0','value_1']] += 1000
        frame.to_csv(path,index=False)
        changed = make_context(configuration(tmp_path,'changed'))
        try:
            np.testing.assert_array_equal(changed.bundle.metadata.static_data['graph_history'],expected)
            torch.testing.assert_close(changed.bundle.val.dataset[0]['X'],source['X'],rtol=0,atol=0)
            assert not torch.equal(changed.bundle.val.dataset[0]['y_raw'],source['y_raw'])
        finally:
            changed.tracker.close()
    finally:
        for context in (first,second,third):
            context.tracker.close()


def test_graph_history_rejects_calendar_channels_and_gaps_but_supports_explicit_contiguous_tail(tmp_path):
    folder = atomic_fixture(tmp_path/'raw',steps=60)
    cfg = configuration(tmp_path,'source')
    cfg['data']['graph'] = dict(history_channel=1)
    with pytest.raises(ValueError,match='observed feature'):
        make_context(cfg)
    path = folder/'Tiny.dyna'
    frame = pd.read_csv(path)
    shifted = pd.to_datetime(frame['time'],utc=True)
    shifted.loc[(frame['dyna_id']%60)>=20] += pd.Timedelta(hours=1)
    frame['time'] = shifted.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.to_csv(path,index=False)
    cfg['data']['graph'] = dict(history_channel=0)
    with pytest.raises(ValueError,match='timestamp gaps'):
        make_context(cfg)
    cfg['data']['graph']['history_steps'] = 8
    context = make_context(cfg)
    try:
        assert context.model.node_features.shape==(8,3)
        assert context.bundle.metadata.input_layout['graph_history']['start_index']==28
    finally:
        context.tracker.close()


def test_static_arrays_are_readonly_and_empty_schema_keeps_existing_checkpoints_compatible():
    values = np.ones((3,2),dtype=np.float32)
    meta = DatasetMetadata('plain',static_data=dict(values=values))
    values[0] = 0
    np.testing.assert_array_equal(meta.static_data['values'],np.ones((3,2)))
    with pytest.raises(ValueError):
        meta.static_data['values'][0] = 0
    plain = SimpleNamespace(metadata=DatasetMetadata('plain'),scaler=NoneScaler())
    assert 'static_data' not in data_state(plain)
    with pytest.raises(TypeError,match='named NumPy arrays'):
        DatasetMetadata('invalid',static_data=dict(values=torch.ones(2)))


def test_graph_prior_objective_cutoff_curriculum_draws_and_prior_only_batches(tmp_path,monkeypatch):
    atomic_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'objective'))
    try:
        batch = next(iter(context.bundle.train))
        prediction = torch.randn_like(batch['y'],requires_grad=True)
        logits = torch.linspace(-1.,1.,9).reshape(3,3).requires_grad_()
        received = []
        def auxiliary(inputs,**kwargs):
            received.append(kwargs)
            return prediction,logits
        monkeypatch.setattr(context.model,'forward_with_aux',auxiliary)
        strategy = {**context.config['training']['strategy'],'graph_regularization_weight':.5}
        step = graph_step(context.loss_fn,strategy)
        state = TrainState(epoch=0,batches_seen=5)
        np.random.seed(37)
        probability = teacher_probability(5,strategy['cl_decay_steps'])
        expected_choices = list(np.random.RandomState(37).uniform(size=2)<probability)
        actual = step(context.model,batch,state)
        base = context.loss_fn(prediction,batch['y'],batch['y_raw'])
        legacy_prior = F.binary_cross_entropy(logits.sigmoid(),context.model.graph_prior)
        torch.testing.assert_close(actual.loss,base.loss+.5*legacy_prior,rtol=1e-6,atol=1e-7)
        assert actual.weight==base.weight and received[-1]['teacher_mask']==expected_choices
        assert state.task_state['graph_regularization_active']
        state.epoch = 1
        torch.testing.assert_close(step(context.model,batch,state).loss,base.loss,rtol=0,atol=0)
        assert not state.task_state['graph_regularization_active']
        invalid = {**batch,'y_raw':torch.full_like(batch['y_raw'],float('nan'))}
        state.epoch = 0
        assert step(context.model,invalid,state).weight==prediction.numel()
        with torch.no_grad():
            logits.fill_(1000)
        assert torch.isfinite(step(context.model,invalid,state).loss)
        state.epoch = 1
        empty = step(context.model,invalid,state)
        assert empty.weight==0 and empty.loss.item()==0
    finally:
        context.tracker.close()


def test_evaluation_prepares_one_graph_per_call_and_never_keeps_a_stale_graph(tmp_path,monkeypatch):
    atomic_fixture(tmp_path/'raw')
    context = make_context(configuration(tmp_path,'evaluate'))
    try:
        args = dict(device=context.device,loss_fn=context.loss_fn,evaluation=context.config['evaluation'])
        expected = evaluate_loop(context.model,context.bundle.test,**args).metrics
        original = context.model.graph_logits
        calls = []
        def counted():
            calls.append(1)
            return original()
        monkeypatch.setattr(context.model,'graph_logits',counted)
        assert evaluate_gts(context.model,context.bundle.test,**args).metrics==expected and len(calls)==1
        with torch.no_grad():
            context.model.fc_cat.bias.copy_(torch.tensor([10.,-10.]))
        calls.clear()
        actual = evaluate_gts(context.model,context.bundle.test,**args).metrics
        assert len(calls)==1
        assert actual==evaluate_loop(context.model,context.bundle.test,**args).metrics
        assert not hasattr(context.model,'prepared_graph')
    finally:
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


def test_full_gumbel_curriculum_regularization_cutoff_bn_adam_schedule_rng_exact_resume(tmp_path):
    periodic_fixture(tmp_path/'raw',name='Tiny',steps=60,nodes=4,features=2,outputs=2)
    trained = run_experiment('train',config=configuration(tmp_path,'full'))
    evaluated = run_experiment('evaluate',config=configuration(tmp_path,'evaluate'),checkpoint=trained.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in trained.metrics.items() if key!='best_validation'}
    partial = make_context(configuration(tmp_path,'partial'))
    initial = partial.model.fc_cat.weight.detach().clone()
    train(partial,until_epoch=1)
    assert not torch.equal(initial,partial.model.fc_cat.weight)
    before = torch.load(partial.directory/'checkpoints/last.pt',weights_only=True)
    assert before['train_state']['task_state']['graph_regularization_active']
    partial.tracker.close()
    resumed = run_experiment('train',config=configuration(tmp_path,'resume'),resume=partial.directory/'checkpoints/last.pt')
    expected,actual = (torch.load(result.artifacts['last'],weights_only=True) for result in (trained,resumed))
    for key in ('model','optimizer','scheduler','amp_scaler','train_state','early_stopper','best_model','rng','loader_rng'):
        equal(expected[key],actual[key])
    assert not actual['train_state']['task_state']['graph_regularization_active']
    assert actual['data_state']['static_data']==dict(graph_history=dict(shape=[36,4],dtype='float32'))
    assert actual['model']['node_features'].shape==(36,4)
    with np.load(evaluated.artifacts['predictions'],allow_pickle=False) as archive:
        assert archive['prediction'].shape[1:]==(2,4,2)


@pytest.mark.parametrize('single_feature_step',(False,True))
def test_actual_cpu_amp_single_node_feature_bn_and_true_tail_batches(tmp_path,single_feature_step):
    atomic_fixture(tmp_path/'raw',steps=46 if single_feature_step else 50,nodes=1)
    cfg = configuration(tmp_path,'amp')
    cfg['data'].update(input_window=1 if single_feature_step else 4,output_window=1)
    if single_feature_step:
        cfg['data']['graph'] = dict(history_steps=1)
        cfg['model']['params']['kernel_size'] = 1
    cfg['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    result = run_experiment('train',config=cfg)
    assert np.isfinite(list(result.metrics.values())).all()


@pytest.mark.parametrize('strategy',[dict(temperature=0),dict(cl_decay_steps=0),dict(graph_regularization_epochs=-1),
    dict(graph_regularization_weight=-1),dict(use_curriculum_learning=1),dict(unknown=1)])
def test_invalid_training_policy_fails_before_data(strategy):
    with pytest.raises(ValueError):
        load_config(config=dict(model=dict(name='GTS'),data=dict(dataset='Tiny'),training=dict(strategy=strategy)))


@pytest.mark.parametrize('params',[dict(kernel_size=0),dict(rnn_units=0),dict(num_rnn_layers=0),
    dict(max_diffusion_step=-1),dict(embedding_dim=0)])
def test_invalid_model_structure_fails_clearly(params):
    with pytest.raises(ValueError,match='GTS'):
        GTS(metadata=metadata(),input_window=4,output_window=2,**params)


def test_invalid_static_history_prior_configuration_and_prediction_shape():
    with pytest.raises(ValueError,match=r'at least 2\*kernel_size-1'):
        GTS(metadata=metadata(points=10),input_window=4,output_window=2)
    meta = metadata(nodes=1)
    invalid = DatasetMetadata('bad',1,2,2,adjacency=np.array([[2.]]),input_layout=meta.input_layout,static_data=meta.static_data)
    with pytest.raises(ValueError,match='probabilities'):
        GTS(metadata=invalid,input_window=4,output_window=2)
    with pytest.raises(ValueError,match='history_steps'):
        load_config(config=dict(model=dict(name='GTS'),data=dict(dataset='Tiny',graph=dict(history_steps=-1))))
    model = GTS(metadata=metadata(),input_window=4,output_window=2,rnn_units=4,kernel_size=3,embedding_dim=6).eval()
    with pytest.raises(ValueError,match='X must'):
        model(dict(X=torch.zeros(1,3,3,2)))
