"""Build actual runtime contexts for update and resume comparisons."""
import logging

from libcity.config.runtime import load_config
from libcity.data.bundle import prepare_traffic
from libcity.executor.train import build_context
from libcity.model.runtime_registry import model_spec, resolve
from libcity.pipeline.experiment import initialize_runtime


def tiny_config(root, name='run', **training):
    return dict(data=dict(dataset='Tiny', data_dir=str(root/'raw'), cache_dir=str(root/'cache'), cache=False,
            input_window=4, output_window=2, scaler={'name':'none'}, pad_training=False),
        model=dict(name='GRU', params={'hidden_size':4}),
        training={'epochs': 3, 'batch_size': 4, 'shuffle': False, **training},
        runtime=dict(device='cpu', seed=17, output_dir=str(root/'runs'), exp_id=name))


def make_context(config):
    config = load_config(config=config)
    device = initialize_runtime(config['runtime'])
    bundle = prepare_traffic(config['data'], config['training'], config['evaluation'], config['runtime']['seed'])
    spec = model_spec(config['model']['name'])
    model = resolve(spec.factory)(metadata=bundle.metadata, input_window=config['data']['input_window'],
        output_window=config['data']['output_window'], **config['model']['params']).to(device)
    directory = config['runtime']['output_dir']
    from pathlib import Path
    directory = Path(directory) / config['runtime']['exp_id']
    directory.mkdir(parents=True, exist_ok=True)
    updates = resolve(spec.updates_per_epoch)(config,bundle) if spec.updates_per_epoch else None
    return build_context(model, bundle, config, device, directory, logging.getLogger('runtime-test'),steps_per_epoch=updates)
