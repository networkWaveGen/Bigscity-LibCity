"""Concentrated regression checks for the modern core, without optional stacks."""

import copy
from importlib import import_module
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

from libcity.config.config_parser import ConfigParser
from libcity.data.batch import Batch, BatchPAD
from libcity.data.utils import BatchCollator, generate_dataloader, generate_dataloader_pad
from libcity.utils.normalization import NoneScaler
from libcity.registry import get_model


@pytest.fixture(scope='module', autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_core_imports_do_not_load_optional_stacks():
    code = '''
import sys
from libcity.pipeline.pipeline import run_model
from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.model.traffic_speed_prediction.RNN import RNN
assert not {'ray', 'dgl', 'torchtext', 'fastdtw', 'gensim'} & set(sys.modules)
'''
    subprocess.run([sys.executable, '-B', '-c', code], check=True)


@pytest.mark.parametrize('padded', [False, True])
def test_dense_batch_values_and_source_ownership(padded):
    samples = [(np.array([[1., 2.]], dtype=np.float64), [3, 4]),
               (np.array([[5., 6.]], dtype=np.float64), [7, 8])]
    original = copy.deepcopy(samples)
    collate = BatchCollator({'X': 'float', 'ids': 'int'}, padded=padded)
    batch = pickle.loads(pickle.dumps(collate))(samples)
    assert isinstance(batch, BatchPAD if padded else Batch)
    batch['X'][0, 0, 0] = 99
    assert samples[0][0][0, 0] == original[0][0][0, 0]
    batch['X'][0, 0, 0] = 1
    batch.to_tensor('cpu')
    torch.testing.assert_close(batch['X'], torch.tensor([[[1., 2.]], [[5., 6.]]]))
    assert batch['ids'].dtype == torch.long
    batch['X'].zero_()
    for actual, expected in zip(samples, original):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
    batch.to_tensor('cpu')  # A second device conversion accepts existing tensors.


def test_padding_and_ragged_fields_do_not_mutate_samples():
    samples = [([1, 2, 3], [4, 5]), ([6], [7])]
    original = copy.deepcopy(samples)
    batch = BatchCollator({'seq': 'int', 'extra': 'no_pad_int'},
                          {'seq': 0}, {'seq': 2})(samples)
    np.testing.assert_array_equal(batch['seq'], [[2, 3], [6, 0]])
    assert batch.get_origin_len('seq') == [2, 1]
    batch.to_tensor('cpu')
    batch['extra'][0][0] = 99
    assert samples == original


def test_padded_loader_keeps_batch_type_without_padding_fields():
    loader, _, _ = generate_dataloader_pad(
        [([1, 2],)], [([1, 2],)], [([1, 2],)], {'ids': 'int'}, 1, 0, None)
    assert isinstance(next(iter(loader)), BatchPAD)


def test_padded_collator_rejects_extra_sample_fields():
    with pytest.raises(KeyError):
        BatchCollator({'ids': 'int'}, {'ids': 0})([([1], 'unexpected')])


def test_loader_works_with_spawn_worker():
    samples = [(np.ones((2, 3, 1)), np.ones((1, 3, 1))) for _ in range(3)]
    loader, _, _ = generate_dataloader(samples, samples, samples,
                                       {'X': 'float', 'y': 'float'}, 2, 1)
    batches = list(loader)
    assert sum(len(batch['X']) for batch in batches) == 3


def test_last_sample_padding_handles_different_input_and_target_shapes():
    samples = [(np.full((4, 2, 1), i), np.full((1, 2, 1), i)) for i in range(3)]
    _, _, loader = generate_dataloader(samples, samples, samples,
                                       {'X': 'float', 'y': 'float'}, 2, 0,
                                       pad_with_last_sample=True)
    batches = list(loader)
    batches[-1].to_tensor('cpu')
    assert batches[-1]['X'].shape == (2, 4, 2, 1)
    assert batches[-1]['y'].shape == (2, 1, 2, 1)
    assert batches[-1]['X'][:, 0, 0, 0].tolist() == [2, 2]


@pytest.mark.parametrize('use_gpu', [False, True])
def test_cpu_device_selection(monkeypatch, use_gpu):
    parser = object.__new__(ConfigParser)
    parser.config = {'gpu': use_gpu}
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'set_device',
                        lambda *_: pytest.fail('CPU path called set_device'))
    parser._init_device()
    assert parser.config['device'] == torch.device('cpu')


def test_config_name_preserves_dotted_stem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'train.v2.json').write_text('{"hidden_size": 17}', encoding='utf-8')
    (tmp_path / 'train.json').write_text('{"hidden_size": 23}', encoding='utf-8')

    parser = object.__new__(ConfigParser)
    parser.config = {}
    parser._parse_config_file('train.v2')

    assert parser['hidden_size'] == 17


def test_packaged_defaults_and_explicit_json_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'raw_data' / 'fixture'
    data.mkdir(parents=True)
    (data / 'config.json').write_text('{}', encoding='utf-8')
    config = tmp_path / 'custom.json'
    config.write_text(json.dumps({'hidden_size': 7}), encoding='utf-8')
    parsed = ConfigParser('traffic_state_pred', 'GRU', 'fixture',
                          config_file=str(config), other_args={'gpu': False})
    assert parsed['hidden_size'] == 7
    assert parsed['model'] == 'RNN'


def test_factory_does_not_hide_constructor_errors(monkeypatch):
    rnn = import_module('libcity.model.traffic_speed_prediction.RNN')

    class Broken:
        def __init__(self, *_):
            raise AttributeError('specific constructor failure')

    monkeypatch.setattr(rnn, 'RNN', Broken)
    with pytest.raises(AttributeError, match='specific constructor failure'):
        get_model({'task': 'traffic_state_pred', 'model': 'RNN'}, {})
