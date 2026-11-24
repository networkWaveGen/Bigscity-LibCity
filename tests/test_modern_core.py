"""Concentrated regression checks for the modern core, without optional stacks."""

from importlib import import_module
import json
import subprocess
import sys

import pytest
import torch

from libcity.config.config_parser import ConfigParser
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
