"""CRANN numerical reference and complete native runtime acceptance."""
import numpy as np
import pandas as pd
import pytest
import torch

from libcity.data.bundle import DatasetMetadata
from libcity.data.dataset.dataset_subclass.crann_dataset import CRANNDataset
from libcity.executor.evaluate import model_inputs
from libcity.executor.train import train_loop
from libcity.model.traffic_flow_prediction.CRANN import CRANN as ReferenceCRANN
from libcity.model.traffic_flow_prediction.runtime_crann import CRANN
from libcity.pipeline.experiment import run_experiment
from libcity.testing.fixtures import atomic_fixture
from runtime_helpers import make_context, tiny_config

pytestmark = pytest.mark.filterwarnings('ignore:Explicit benchmark protocol changes')


def configuration(root, experiment):
    config = tiny_config(root, experiment, epochs=2, batch_size=8, shuffle=True)
    config['model'] = dict(name='CRANN', params=dict(dim_x=1, dim_y=3, n_hidden_tem=4, n_ar=2))
    config['data'].update(cache=True, input_window=4, output_window=2,
        sampling=dict(name='crann', temporal_history=6), scaler={'name': 'standard'},
        features=dict(load_external=True, known_future_columns=['holiday']))
    config['training']['scheduler'] = dict(name='step', interval='epoch', step_size=1, gamma=.7)
    return config


def fixture(root):
    folder = atomic_fixture(root/'raw', steps=100)
    times = pd.read_csv(folder/'Tiny.dyna')['time'].drop_duplicates().to_numpy()
    pd.DataFrame(dict(ext_id=range(100), time=times, temperature=np.arange(100)*.01,
                      weather=np.arange(100)*.1, holiday=np.arange(100)%2)).to_csv(folder/'Tiny.ext', index=False)
    return folder


def test_reference_forward_and_gradient_match_for_original_horizon():
    metadata = DatasetMetadata('Reference', 3, 3, 1, np.eye(3), {},
        dict(kind='crann', temporal_history=6, length=6, future_feature_indices=[2]))
    args = dict(dim_x=1, dim_y=3, n_hidden_tem=4, n_ar=2, input_window=4, output_window=24)
    native = CRANN(metadata=metadata, **args).eval()
    reference = ReferenceCRANN(dict(args, device=torch.device('cpu')),
        dict(num_nodes=3, feature_dim=1, output_dim=1, ext_dim=1)).eval()
    native.load_state_dict(reference.state_dict())
    batch = dict(X=torch.randn(2, 6, 3, 3), future_features=torch.randn(2, 24, 3, 1))
    old_batch = dict(x_time=batch['X'][:, :, 0, 1:2],
        x_space=batch['X'][:, -4:, :, 0].reshape(2, 4, 1, 3), x_ext=batch['future_features'][:, :, 0])
    expected = reference(old_batch).reshape(2, 24, 3, 1)
    actual = native(batch)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for name, parameter in native.named_parameters():
        target = dict(reference.named_parameters())[name]
        if target.grad is not None:
            torch.testing.assert_close(parameter.grad, target.grad, rtol=1e-5, atol=1e-6)


def test_crann_native_inputs_raw_mask_single_batch_and_target_time_split(tmp_path):
    fixture(tmp_path)
    context = make_context(configuration(tmp_path, 'inspect'))
    try:
        bundle = context.bundle
        intervals = [set((loader.dataset.target_starts[:, None]+np.arange(2)).ravel())
                     for loader in (bundle.train, bundle.val, bundle.test)]
        assert not intervals[0]&intervals[1] and not intervals[1]&intervals[2]
        batch = next(iter(bundle.train))
        inputs = {key: value[:1] for key, value in model_inputs(batch, 1).items()}
        context.model.eval()
        prediction = context.model(inputs)
        assert prediction.shape == (1, 2, 3, 1)
        poisoned = {**inputs, 'y': torch.full_like(batch['y'][:1], float('nan'))}
        torch.testing.assert_close(prediction, context.model(poisoned))
        for module in context.model.modules():
            assert not any(hasattr(module, key) for key in ('_scaler', '_logger', 'device'))
        assert bundle.metadata.input_layout['length'] == 6
        assert bundle.metadata.input_layout['future_feature_indices'] == (3,)
        assert sum(int((item['y_raw'] == 0).sum()) for item in bundle.train) > 0
        for item in bundle.train:
            result = context.loss_fn(item['y']+1e-5, item['y']+1e-5, item['y_raw'])
            assert result.weight == int((item['y_raw'] != 0).sum())
    finally:
        context.tracker.close()


def test_crann_short_training_independent_evaluation_and_exact_resume(tmp_path):
    fixture(tmp_path)
    full = run_experiment('train', config=configuration(tmp_path, 'full'))
    evaluated = run_experiment('evaluate', config=configuration(tmp_path, 'evaluate'),
                               checkpoint=full.artifacts['best'])
    assert full.metrics['loss'] == evaluated.metrics['loss']
    expected = torch.load(full.artifacts['last'], weights_only=True)
    partial = make_context(configuration(tmp_path, 'partial'))
    train_loop(partial, until_epoch=1)
    partial.tracker.close()
    resumed = run_experiment('train', config=configuration(tmp_path, 'resumed'),
                             resume=partial.directory/'checkpoints/last.pt')
    actual = torch.load(resumed.artifacts['last'], weights_only=True)
    assert expected['train_state'] == actual['train_state']
    assert expected['scheduler'] == actual['scheduler']
    for name, value in expected['model'].items():
        torch.testing.assert_close(value, actual['model'][name], rtol=0, atol=0)


def test_reference_dataset_preserves_raw_targets_before_scaling(tmp_path):
    fixture(tmp_path)
    dataset = CRANNDataset(dict(dataset='Tiny', data_dir=str(tmp_path/'raw'), cache_dir=str(tmp_path/'cache'),
        cache_dataset=False, n_timesteps=6, input_window=4, output_window=2, dim_x=1, dim_y=3,
        scaler='standard', train_rate=.6, eval_rate=.2, batch_size=8, data_col=['speed'],
        pad_with_last_sample=False))
    train, _, _ = dataset.get_data()
    batches = list(train)
    assert sum(int((batch['y_raw'] == 0).sum()) for batch in batches) > 0
    for batch in batches:
        torch.testing.assert_close(dataset.scaler.inverse_transform(batch['y']), batch['y_raw'])
