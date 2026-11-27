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
    spec = model_spec(config['model']['name'])
    adapter=resolve(spec.prepare_data) if spec.prepare_data else prepare_traffic
    bundle = adapter(config['data'], config['training'], config['evaluation'], config['runtime']['seed'])
    arguments=dict(metadata=bundle.metadata,**config['model']['params'])
    if spec.task=='traffic_state_pred':
        arguments.update(input_window=config['data']['input_window'],output_window=config['data']['output_window'])
    model = resolve(spec.factory)(**arguments).to(device)
    directory = config['runtime']['output_dir']
    from pathlib import Path
    directory = Path(directory) / config['runtime']['exp_id']
    directory.mkdir(parents=True, exist_ok=True)
    updates = resolve(spec.updates_per_epoch)(config,bundle) if spec.updates_per_epoch else None
    loss_fn=resolve(spec.loss)(bundle,config['training']['loss']) if spec.loss else None
    evaluator=resolve(spec.evaluate_loader) if spec.evaluate_loader else None
    return build_context(model, bundle, config, device, directory, logging.getLogger('runtime-test'),steps_per_epoch=updates,
        loss_fn=loss_fn,evaluate_fn=evaluator)
