"""Numerical regressions for retained source, not extra runtime support."""
import copy
import importlib
import logging
import numpy as np
import pytest
import torch
from libcity.data.dataset.traffic_state_datatset import TrafficStateDataset
from libcity.data.dataset.dataset_subclass.crann_dataset import CRANNDataset
from libcity.data.utils import generate_dataloader
from libcity.model.traffic_od_prediction.CSTN import CSTN
from libcity.utils.checkpoint import copy_state_to_cpu
from libcity.utils.normalization import NoneScaler


@pytest.fixture(autouse=True)
def isolated_run(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference_model(config,features):
    paths={'STNN':'traffic_flow_prediction.STNN','DSAN':'traffic_flow_prediction.DSAN'}
    cls=getattr(importlib.import_module('libcity.model.'+paths[config['model']]),config['model'])
    return cls(config,features)


@pytest.mark.parametrize('mode', [None, 'refine', 'discover'])
def test_stnn_forecasts_on_cpu(monkeypatch, mode):
    monkeypatch.setattr(torch, 'set_default_tensor_type',
                        lambda *_: pytest.fail('STNN changed the global tensor type'))
    device = torch.device('cpu')
    config = {'task': 'traffic_state_pred', 'model': 'STNN', 'device': device,
              'input_window': 12, 'output_window': 2, 'mode': mode}
    model = reference_model(config, {
        'scaler': NoneScaler(), 'num_nodes': 41,
        'adj_mx': np.roll(np.eye(41, dtype=np.float32), 1, axis=1),
    })
    batch = {'X': torch.rand(2, 12, 41, 1, device=device),
             'y': torch.ones(2, 2, 41, 1, device=device)}

    prediction = model.predict(batch)
    assert prediction.shape == batch['y'].shape
    assert prediction.device == model.relations.device == device
    assert torch.isfinite(prediction).all()
    prediction.mean().backward()
    assert model.factors.grad is not None


def test_dsan_forecasts_and_trains_on_cpu():
    device = torch.device('cpu')
    config = {'task': 'traffic_state_pred', 'model': 'DSAN', 'device': device,
              'input_window': 2, 'output_window': 2, 'L': 1,
              'l_d': 1, 'r_d': 0}
    model = reference_model(config, {
        'scaler': NoneScaler(), 'len_row': 2, 'len_column': 2, 'num_nodes': 4,
        'feature_dim': 3, 'ext_dim': 1, 'output_dim': 2,
    })
    batch = {'X': torch.rand(2, 2, 2, 2, 3, device=device),
             'y': torch.ones(2, 2, 2, 2, 2, device=device)}

    prediction = model.predict(batch)
    assert prediction.shape == batch['y'].shape
    assert prediction.device == device
    loss = model.calculate_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.dsan.final_layer.weight.grad is not None


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


@pytest.mark.parametrize('count', [1, 2, 3])
def test_cstn_actual_batch_size_and_gradient(count):
    model = CSTN(dict(device=torch.device('cpu'), batch_size=2, input_window=1,
                      output_window=1, output_dim=1, n_layers=1, c_lt=2),
                 dict(scaler=NoneScaler(), len_row=1, len_column=1))
    batch = dict(X=torch.ones(count, 1, 1, 1, 1, 1, 1), W=torch.ones(count, 1, 29),
                 y=torch.ones(count, 1, 1, 1, 1, 1, 1))
    output = model.predict(batch)
    assert output.shape == batch['y'].shape
    loss = model.calculate_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('cache', [False, True])
def test_real_dataset_get_data_twice_preserves_loaders_values_and_scaler(cache, tmp_path):
    runner = object.__new__(TrafficStateDataset)
    runner.data = None
    runner.cache_dataset = cache
    runner.cache_file_name = str(tmp_path / 'cached.npz')
    arrays = [np.arange(size, dtype=float).reshape(size, 1, 1, 1) + 1 for size in (6, 6, 2, 2, 2, 2)]
    if cache:
        np.savez(runner.cache_file_name, cache_config=np.asarray(runner._cache_config_json()), **dict(zip(
            ('x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test'), arrays)))
    calls = []

    def generate():
        calls.append(True)
        return tuple(value.copy() for value in arrays)

    runner._generate_train_val_test = generate
    runner.robustness_test = False
    runner.scaler_type, runner.ext_scaler_type = 'standard', 'none'
    runner.output_dim, runner.normal_external = 1, False
    runner.feature_name = {'X': 'float', 'y': 'float'}
    runner.batch_size, runner.num_workers, runner.pad_with_last_sample = 2, 0, False
    runner._logger = logging.getLogger(__name__)
    first = runner.get_data()
    mean, std = runner.scaler.mean, runner.scaler.std
    before = [copy.deepcopy(loader.dataset[0]) for loader in first]
    second = runner.get_data()
    assert all(a is b for a, b in zip(first, second))
    assert (runner.scaler.mean, runner.scaler.std) == (mean, std)
    for loader, original in zip(second, before):
        np.testing.assert_array_equal(loader.dataset[0][0], original[0])
    assert len(calls) == (0 if cache else 1)
