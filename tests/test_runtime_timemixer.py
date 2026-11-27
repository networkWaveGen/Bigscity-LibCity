"""All forecasting branches use time-axis decomposition and local RevIN state."""
import json
import math

import pandas as pd
import pytest
import torch

from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_flow_prediction.TimeMixer import TimeMixer as ReferenceTimeMixer
from libcity.model.traffic_flow_prediction.runtime_timemixer import FrequencyDecomposition,Normalize
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from libcity.utils.normalization import NoneScaler
from runtime_helpers import make_context,tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root,experiment,independent=False):
    config = tiny_config(root,experiment,epochs=2,batch_size=8,shuffle=True)
    config['data'].update(input_window=8,output_window=3)
    config['model'] = dict(name='TimeMixer',params=dict(d_model=4,d_ff=8,e_layers=1,moving_avg=3,
        down_sampling_layers=1,channel_independence=independent,dropout=.1))
    config['training']['scheduler'] = dict(name='step',interval='epoch',step_size=1,gamma=.8)
    return config


REFERENCE_CASES = [(method,independent,normalized) for method in ('avg','max','conv')
                   for independent,normalized in ((False,False),(True,False),(True,True))]


@pytest.mark.parametrize('method,independent,normalized',REFERENCE_CASES)
def test_original_supported_branches_forecast_and_gradients(tmp_path,method,independent,normalized):
    atomic_fixture(tmp_path/'raw',steps=90)
    config = configuration(tmp_path,'reference',independent)
    config['model']['params'].update(down_sampling_method=method,use_norm=normalized,dropout=0.)
    context = make_context(config)
    try:
        params = context.config['model']['params']
        reference = ReferenceTimeMixer(dict(params,input_window=8,output_window=3,
            channel_independence=int(independent),use_norm=int(normalized)),dict(num_nodes=3,feature_dim=1,scaler=NoneScaler())).eval()
        model = context.model.eval()
        current_keys = model.state_dict().keys()
        weights = reference.state_dict()
        removed = set(weights)-set(current_keys)
        assert all(key.startswith(('enc_embedding.position_embedding.','enc_embedding.temporal_embedding.'))
            or '.layer_norm.' in key or (not independent and '.out_cross_layer.' in key)
            or (independent and key.startswith('out_layer.')) for key in removed)
        model.load_state_dict({key:value for key,value in weights.items() if key in current_keys})
        batch = next(iter(context.bundle.train))
        expected = reference.predict(batch)
        if independent:
            expected = expected.unsqueeze(-1)
        actual = model(model_inputs(batch,1))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for name,parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad,dict(reference.named_parameters())[name].grad,rtol=1e-4,atol=1e-6)
        assert all(not any(hasattr(module,key) for key in ('device','_scaler','_logger','mean','stdev')) for module in model.modules())
    finally:
        context.tracker.close()


@pytest.mark.parametrize('independent',[False,True])
def test_full_lifecycle_and_exact_dropout_resume(tmp_path,independent):
    atomic_fixture(tmp_path/'raw',steps=90)
    def config(experiment):
        value = configuration(tmp_path,experiment,independent)
        value['model']['params'].update(use_norm=True,down_sampling_method='conv',decomp_method='dft_decomp')
        return value
    full = run_experiment('train',config=config('full'))
    evaluated = run_experiment('evaluate',config=config('evaluate'),checkpoint=full.artifacts['best'])
    assert evaluated.metrics=={key:value for key,value in full.metrics.items() if key!='best_validation'}
    expected = torch.load(full.artifacts['last'],weights_only=True)
    partial = make_context(config('partial'))
    train_loop(partial,until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train',config=config('resume'),resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'],weights_only=True)
    assert actual['train_state']==expected['train_state']
    assert actual['scheduler']==expected['scheduler']
    for key,value in expected['model'].items():
        torch.testing.assert_close(value,actual['model'][key],rtol=0,atol=0)


@pytest.mark.parametrize('independent',[False,True])
def test_multitarget_odd_scales_even_kernel_float64_and_no_mutation(tmp_path,independent):
    folder = atomic_fixture(tmp_path/'raw',steps=90)
    frame = pd.read_csv(folder/'Tiny.dyna')
    frame['flow'],frame['observation'] = frame['speed']*.5+2,frame['speed']*.1
    frame.to_csv(folder/'Tiny.dyna',index=False)
    path = folder/'config.json';metadata = json.loads(path.read_text())
    metadata.update(data_col=['speed','flow','observation'],output_dim=2)
    path.write_text(json.dumps(metadata))
    config = configuration(tmp_path,'multivariate',independent)
    config['data']['input_window'] = 7
    config['model']['params'].update(d_model=5,moving_avg=4,down_sampling_method='conv',down_sampling_layers=2,use_norm=True)
    context = make_context(config)
    try:
        model = context.model.eval().double()
        assert model.scale_lengths==(7,4,2)
        batch = next(iter(context.bundle.test))
        inputs = {key:value[:1].double() for key,value in model_inputs(batch,2).items()}
        original = inputs['X'].clone()
        states = {key:value.clone() for key,value in model.state_dict().items()}
        rng = torch.get_rng_state()
        prediction = model(inputs)
        assert prediction.shape==(1,3,3,2) and prediction.dtype==torch.float64
        assert torch.isfinite(prediction).all()
        model({'X':inputs['X']*4+3})
        torch.testing.assert_close(prediction,model({**inputs,'y':torch.full_like(prediction,float('nan'))}),rtol=0,atol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        torch.testing.assert_close(original,inputs['X'],rtol=0,atol=0)
        for key,value in states.items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        context.tracker.close()


def test_frequency_decomposition_uses_time_axis_per_sample_and_requested_top_k():
    time = torch.arange(9,dtype=torch.float64)
    fundamental = torch.sin(2*math.pi*time/9)
    harmonic = .3*torch.cos(2*math.pi*3*time/9)
    values = torch.stack((3+fundamental+harmonic,5+2*fundamental+harmonic))[:,:,None].repeat(1,1,3)
    source = values.clone().requires_grad_()
    seasonal,trend = FrequencyDecomposition(1)(source)
    expected = torch.stack((fundamental,2*fundamental))[:,:,None].expand_as(source)
    torch.testing.assert_close(seasonal,expected,rtol=1e-8,atol=1e-8)
    torch.testing.assert_close(seasonal+trend,source,rtol=0,atol=1e-14)
    both,_ = FrequencyDecomposition(2)(source)
    torch.testing.assert_close(both,source-source.mean(1,keepdim=True),rtol=1e-8,atol=1e-8)
    seasonal.square().mean().backward()
    assert torch.isfinite(source.grad).all() and source.grad.abs().sum()>0
    short = torch.randn(1,1,2)
    torch.testing.assert_close(FrequencyDecomposition(5)(short)[0],torch.zeros_like(short))


def test_normalization_context_is_explicit_and_does_not_leak_between_calls():
    layer = Normalize(3,True).double()
    source = torch.randn(2,5,3,dtype=torch.float64)
    normalized,state = layer.normalize(source)
    layer.normalize(source*100)
    torch.testing.assert_close(layer.denormalize(normalized,state),source,rtol=1e-8,atol=1e-8)
    expanded = normalized.unsqueeze(-1).expand(-1,-1,-1,2)
    torch.testing.assert_close(layer.denormalize(expanded,state),source.unsqueeze(-1).expand_as(expanded),rtol=1e-8,atol=1e-8)
    assert not hasattr(layer,'mean') and not hasattr(layer,'stdev')


@pytest.mark.parametrize('independent',[False,True])
def test_actual_cpu_amp_fft_training(tmp_path,independent):
    atomic_fixture(tmp_path/'raw',steps=90)
    config = configuration(tmp_path,'amp',independent)
    config['model']['params'].update(decomp_method='dft_decomp',use_norm=True,d_model=5)
    config['training'].update(epochs=1,amp=True,amp_dtype='bfloat16')
    assert run_experiment('train',config=config).metrics['loss']>=0


def test_zero_scale_layers_and_excessive_pooling_guard(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=90)
    config = configuration(tmp_path,'single-scale')
    config['model']['params']['down_sampling_layers'] = 0
    assert run_experiment('train',config=config).metrics['loss']>=0
    config['model']['params']['down_sampling_layers'] = 4
    with pytest.raises(ValueError,match='empty scale'):
        make_context(config)
