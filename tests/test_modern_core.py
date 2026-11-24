"""Concentrated regression checks for the modern core, without optional stacks."""

import json

import pytest
import torch

from libcity.config import ConfigParser


@pytest.fixture(scope='module', autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


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
