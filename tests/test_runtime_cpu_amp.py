"""Actual CPU convolution AMP training, backend restoration and exact resume."""
import torch

from libcity.executor.train import train_loop,cpu_amp_backward_supported
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from runtime_helpers import make_context, tiny_config


def test_cpu_convolution_amp_lifecycle_and_backend_restoration(tmp_path):
    atomic_fixture(tmp_path/'raw', steps=70)
    def config(name):
        value = tiny_config(tmp_path, name, epochs=2, amp=True, amp_dtype='bfloat16', batch_size=8, shuffle=True)
        value['model'] = dict(name='STGCN', params=dict(Ks=2, Kt=2, blocks=[[1,4,4],[4,4,4]], dropout=.1))
        value['data']['input_window'] = 12
        return value
    backend = torch.backends.mkldnn.enabled
    full = run_experiment('train', config=config('full'))
    assert torch.backends.mkldnn.enabled == backend
    expected = torch.load(full.artifacts['last'], weights_only=True)
    partial = make_context(config('partial'))
    train_loop(partial, until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train', config=config('resume'), resume=partial.directory/'checkpoints/last.pt')
    assert torch.backends.mkldnn.enabled == backend
    actual = torch.load(resumed.artifacts['last'], weights_only=True)
    assert expected['train_state'] == actual['train_state']
    for name, value in expected['model'].items():
        torch.testing.assert_close(value, actual['model'][name], rtol=0, atol=0)


def test_cpu_amp_fallback_does_not_require_a_convolution_module(tmp_path):
    atomic_fixture(tmp_path/'raw',steps=60)
    config = tiny_config(tmp_path,'linear',epochs=1,amp=True)
    config['model'] = dict(name='STAEformer',params=dict(input_embedding_dim=4,tod_embedding_dim=2,
        dow_embedding_dim=2,spatial_embedding_dim=2,adaptive_embedding_dim=2,
        feed_forward_dim=16,num_heads=2,num_layers=1,dropout=.1))
    context = make_context(config)
    try:
        assert not any(isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d))
                       for module in context.model.modules())
        assert context.native_cpu_amp_backward==(torch.backends.mkldnn.enabled and not cpu_amp_backward_supported())
        backend = torch.backends.mkldnn.enabled
        assert train_loop(context)>=0
        assert torch.backends.mkldnn.enabled==backend
    finally:
        context.tracker.close()
