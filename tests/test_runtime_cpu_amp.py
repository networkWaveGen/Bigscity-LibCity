"""Actual CPU convolution AMP training, backend restoration and exact resume."""
import torch

from libcity.executor.train import train_loop
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
