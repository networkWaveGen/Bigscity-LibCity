"""Regression coverage for evaluation, checkpoint, configuration and CPU fixes."""

import copy
import importlib
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from libcity.config.config_parser import ConfigParser
from libcity.data.dataset.dataset_subclass.crann_dataset import CRANNDataset
from libcity.data.dataset.traffic_state_datatset import TrafficStateDataset
from libcity.data.utils import generate_dataloader
from libcity.executor.chebconv_executor import ChebConvExecutor
from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.utils.checkpoint import copy_state_to_cpu
from libcity.utils.normalization import NoneScaler


@pytest.fixture(autouse=True)
def isolated_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Forecast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))

    def predict(self, batch):
        return torch.zeros_like(batch['y']) + self.weight

    def calculate_loss(self, batch):
        return (self.predict(batch) - batch['y']).abs().mean()


def executor(executor_class=TrafficStateExecutor, **overrides):
    config = dict(evaluator='TrafficStateEvaluator', metrics=['MAE'],
                  evaluator_mode='single', save_mode=['csv'], device=torch.device('cpu'),
                  exp_id=1, model='Forecast', dataset='fixture', max_epoch=2,
                  learning_rate=0.1, saved_model=False, load_best_epoch=True,
                  save_predictions=True)
    config.update(overrides)
    return executor_class(config, Forecast(), {'scaler': NoneScaler()})


def samples():
    return [(np.zeros((1, 1, 1)), np.full((1, 1, 1), value)) for value in (1., 2., 100.)]


@pytest.mark.parametrize('pad', [False, True])
def test_evaluation_and_validation_use_only_real_samples(tmp_path, pad):
    train, valid, test = generate_dataloader(samples(), samples(), samples(),
        {'X': 'float', 'y': 'float'}, 2, 0, pad_with_last_sample=pad)
    assert sum(len(batch['X']) for batch in train) == (4 if pad else 3)
    assert [value for batch in valid for value in batch['y'].reshape(-1)] == [1., 2., 100.]
    runner = executor()
    try:
        assert runner._valid_epoch(valid, 0) == pytest.approx(103 / 3)
        result = runner.evaluate(test)
        assert result['MAE'].iloc[0] == pytest.approx(103 / 3)
        archive = next((tmp_path / 'libcity/cache/1/evaluate_cache').glob('*.npz'))
        with np.load(archive) as stored:
            assert stored['truth'].shape[0] == 3
            np.testing.assert_array_equal(stored['truth'].reshape(-1), [1., 2., 100.])
    finally:
        runner._writer.close()


def test_no_disk_save_restores_best_model_and_nonempty_adam(tmp_path):
    train, valid, _ = generate_dataloader(samples(), samples(), samples(),
        {'X': 'float', 'y': 'float'}, 2, 0, shuffle=False)
    runner = executor()
    states = []

    def validation(*_):
        states.append((copy.deepcopy(runner.model.state_dict()),
                       copy.deepcopy(runner.optimizer.state_dict())))
        return (0.1, 1.)[len(states) - 1]

    runner._valid_epoch = validation
    try:
        runner.train(train, valid)
        torch.testing.assert_close(runner.model.weight, states[0][0]['weight'])
        restored = runner.optimizer.state_dict()['state']
        assert restored
        for key in restored:
            for field in restored[key]:
                torch.testing.assert_close(restored[key][field], states[0][1]['state'][key][field])
        assert not list((tmp_path / 'libcity/cache/1/model_cache').iterdir())
        assert states[0][0]['weight'].item() != states[1][0]['weight'].item()
    finally:
        runner._writer.close()


def test_chebconv_without_disk_save_keeps_existing_checkpoint(tmp_path):
    runner = executor(ChebConvExecutor, max_epoch=1)
    checkpoint = tmp_path / runner.cache_dir / 'previous-run.tar'
    checkpoint.write_bytes(b'previous checkpoint')
    runner._train_epoch = lambda *_: 0.
    runner._valid_epoch = lambda *_: 0.1
    try:
        runner.train([], [])
        assert list(checkpoint.parent.iterdir()) == [checkpoint]
        assert checkpoint.read_bytes() == b'previous checkpoint'
    finally:
        runner._writer.close()


def test_cpu_snapshot_owns_optimizer_tensors_and_state_metadata():
    model = torch.nn.Linear(1, 1)
    state = model.state_dict()
    saved = copy_state_to_cpu(state)
    assert saved._metadata == state._metadata
    before = saved['weight'].clone()
    with torch.no_grad():
        model.weight.add_(1)
    torch.testing.assert_close(saved['weight'], before)
    assert all(tensor.device.type == 'cpu' for tensor in saved.values())


@pytest.mark.parametrize(('train_rate', 'eval_rate'), [
    (1., 0.), (.9, .2), (-.1, .2), (float('nan'), .2), (.7, 0.),
])
def test_invalid_split_ratios_fail(train_rate, eval_rate):
    dataset = object.__new__(TrafficStateDataset)
    dataset.train_rate, dataset.eval_rate = train_rate, eval_rate
    dataset.cache_dataset = False
    data = np.arange(10).reshape(10, 1, 1, 1)
    with pytest.raises(ValueError, match='train_rate|empty'):
        dataset._split_train_val_test(data, data.copy())


@pytest.mark.parametrize(('dataset_class', 'field_count'), [
    (TrafficStateDataset, 2),
    (CRANNDataset, 4),
])
def test_time_splits_reject_empty_test_and_keep_fields_aligned(dataset_class, field_count, tmp_path):
    dataset = object.__new__(dataset_class)
    dataset.train_rate, dataset.eval_rate = .6, .2
    dataset._logger = logging.getLogger(__name__)
    dataset.cache_dataset = True
    cache_file = tmp_path / 'split.npz'
    dataset.cache_file_folder = tmp_path
    dataset.cache_file_name = str(cache_file)
    data = np.arange(10).reshape(10, 1, 1, 1)
    fields = [data + 100 * index for index in range(field_count)]

    with pytest.raises(ValueError, match='empty train/validation/test splits'):
        dataset._split_train_val_test(*(field[:2] for field in fields))
    assert not cache_file.exists()

    actual = dataset._split_train_val_test(*fields)
    assert cache_file.is_file()
    expected = [field[start:stop] for start, stop in ((0, 6), (6, 8), (8, 10)) for field in fields]
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)


@pytest.mark.parametrize('external', [{}, {'learning_rate': .1}])
def test_explicit_arguments_override_search_but_search_overrides_json_defaults(external):
    config = object.__new__(ConfigParser)
    config.config = {}
    config._parse_external_config('traffic_state_pred', 'GRU', 'fixture',
                                 other_args=external, hyper_config_dict={'learning_rate': .001})
    assert config['learning_rate'] == external.get('learning_rate', .001)


@pytest.mark.parametrize(('train', 'cached', 'task'), [
    (False, False, 'traffic_state_pred'), (False, True, 'traffic_state_pred'),
    (True, False, 'traffic_state_pred'), (False, False, 'map_matching'),
])
def test_pipeline_uses_resolved_mode_and_never_silently_trains(monkeypatch, train, cached, task):
    pipeline = importlib.import_module('libcity.pipeline.pipeline')
    class Config(dict):
        @property
        def config(self):
            return self

    settings = Config(task=task, train=train, saved_model=False, exp_id=1, seed=0)
    calls = []
    monkeypatch.setattr(pipeline, 'ConfigParser', lambda *args: settings)
    monkeypatch.setattr(pipeline.os.path, 'isfile', lambda *_: cached)
    monkeypatch.setattr(pipeline, 'get_logger', lambda *_: logging.getLogger(__name__))
    monkeypatch.setattr(pipeline, 'set_random_seed', lambda *_: None)
    monkeypatch.setattr(pipeline, 'get_dataset', lambda *_: SimpleNamespace(
        get_data=lambda: ([], [], []), get_data_feature=lambda: {}))
    monkeypatch.setattr(pipeline, 'get_model', lambda *_: Forecast())
    monkeypatch.setattr(pipeline, 'get_executor', lambda *_: SimpleNamespace(
        train=lambda *_: calls.append('train'), save_model=lambda *_: calls.append('save'),
        load_model=lambda *_: calls.append('load'), evaluate=lambda *_: calls.append('evaluate')))
    if not train and not cached and task != 'map_matching':
        with pytest.raises(FileNotFoundError, match='Evaluation-only mode'):
            pipeline.run_model(task, 'GRU', 'fixture', train=True, saved_model=True)
        assert calls == []
    else:
        pipeline.run_model(task, 'GRU', 'fixture', train=True, saved_model=True)
        assert calls == (['train', 'evaluate'] if train else (
            ['load', 'evaluate'] if task != 'map_matching' else ['evaluate']))


@pytest.mark.parametrize('name', ['ACFM', 'ACFMCommon'])
@pytest.mark.parametrize('gru', [False, True])
def test_convolution_recurrent_states_follow_cpu_and_dtype(name, gru):
    module = importlib.import_module('libcity.model.traffic_flow_prediction.' + name)
    network = module.ConvLSTM(1, 2, 2, lstm_channels=2, dropout_rate=0, gru=gru).double()
    value = torch.randn(2, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    output = network(value)
    assert output.device == value.device and output.dtype == value.dtype
    output.square().mean().backward()
    assert torch.isfinite(value.grad).all()


def test_dstagnn_constructs_forecasts_and_backpropagates_on_cpu():
    from libcity.model.traffic_flow_prediction.DSTAGNN import DSTAGNN
    adjacency = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
    model = DSTAGNN(dict(device=torch.device('cpu'), input_window=12, output_window=2,
                        nb_block=1, K=2, nb_chev_filter=4, nb_time_filter=4,
                        d_model=4, d_k=2, n_heads=1),
                    dict(scaler=NoneScaler(), adj_mx=adjacency, adj_TMD=adjacency,
                         adj_pa=adjacency, num_nodes=4))
    batch = {'X': torch.rand(2, 12, 4, 1), 'y': torch.ones(2, 2, 4, 1)}
    assert model.predict(batch).shape == batch['y'].shape
    value = model.calculate_loss(batch)
    value.backward()
    assert torch.isfinite(value)
    assert model.BlockList[0].adj_pa.device.type == 'cpu'
    assert 'BlockList.0.adj_pa' not in model.state_dict()


def test_stssl_temporal_loss_accepts_real_short_cpu_batch():
    from libcity.model.traffic_flow_prediction.STSSL import TemporalHeteroModel
    model = TemporalHeteroModel(4, batch_size=8, num_nodes=3, device=torch.device('cpu'))
    z1 = torch.randn(2, 1, 3, 4, requires_grad=True)
    value = model(z1, torch.randn_like(z1))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(z1.grad).all()


def test_stssl_graph_is_a_device_following_nonpersistent_buffer():
    from libcity.model.traffic_flow_prediction.STSSL import STSSL
    model = STSSL(dict(device=torch.device('cpu'), d_model=8, batch_size=4),
                  dict(scaler=NoneScaler(), num_nodes=4, len_closeness=12, len_period=0,
                       adj_mx=np.ones((4, 4), dtype=np.float32)))
    model.double()
    assert model.graph.device.type == 'cpu' and model.graph.dtype == torch.float64
    assert 'graph' not in model.state_dict()


def test_sstban_temporal_embedding_and_masked_attention_stay_on_cpu():
    from libcity.model.traffic_flow_prediction.SSTBAN import STEmbedding, MAB
    embedding = STEmbedding(11, 4, .1, gpu=1)
    spatial = torch.randn(3, 4)
    temporal = torch.zeros((2, 4, 2))
    encoded = embedding(spatial, temporal, 4)
    assert encoded.shape == (2, 4, 3, 4)
    attention = MAB(1, 4, 4, 4, .1)
    query = torch.randn(2, 4, 3, 4, requires_grad=True)
    value = attention(query, query, 2, mask=torch.ones_like(query))
    assert value.device.type == 'cpu' and torch.isfinite(value).all()
    value.square().mean().backward()
    assert torch.isfinite(query.grad).all()


def test_rgsl_graph_forecast_and_gradient_on_cpu():
    from libcity.model.traffic_flow_prediction.RGSL import RGSL
    torch.manual_seed(17)
    adjacency = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
    model = RGSL(dict(device=torch.device('cpu'), output_window=2, rnn_units=4,
                      num_layers=1, embed_dim=4),
                 dict(scaler=NoneScaler(), num_nodes=4, adj_mx=adjacency))
    batch = {'X': torch.rand(2, 3, 4, 1), 'y': torch.ones(2, 2, 4, 1)}
    assert model.predict(batch).shape == batch['y'].shape
    loss = model.calculate_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.node_embeddings.grad is not None
