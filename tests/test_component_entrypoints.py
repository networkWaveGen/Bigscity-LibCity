"""Check actual factories across all tasks, beyond traffic forecasting."""

import json

import numpy as np
import pytest
import torch

from libcity.config.config_parser import ConfigParser
from libcity.data.utils import get_dataset
from libcity.utils.normalization import NoneScaler
from libcity.registry import get_evaluator, get_executor, get_model


@pytest.fixture(autouse=True)
def tiny_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'raw_data/fixture'
    data.mkdir(parents=True)
    (data / 'config.json').write_text('{}', encoding='utf-8')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config_for(task, model):
    return ConfigParser(task, model, 'fixture', other_args=dict(
        gpu=False, exp_id=123, hidden_size=8, loc_emb_size=4, tim_emb_size=4,
        dropout_p=0, lr_decay=False, num_nodes=3, max_epoch=1))


def features_for(task):
    return dict(num_nodes=3, num_edges=3, feature_dim=1, output_dim=1,
                scaler=NoneScaler(), adj_mx=np.ones((3, 3), dtype=np.float32),
                loc_size=6, tim_size=49, loc_pad=5, tim_pad=48, uid_size=3)


@pytest.mark.parametrize(('task', 'model_name'), [
    ('traffic_state_pred', 'GRU'), ('traj_loc_pred', 'GRU'),
    ('eta', 'DeepTTE'), ('map_matching', 'HMMM'), ('road_representation', 'LINE'),
])
def test_real_model_executor_and_evaluator_factories(task, model_name):
    config = config_for(task, model_name)
    feature = features_for(task)
    model = get_model(config, feature)
    executor = get_executor(config, model, feature)
    evaluator = get_evaluator(config)
    assert executor.model is model
    assert type(executor.evaluator) is type(evaluator)
    if hasattr(executor, '_writer'):
        executor._writer.close()


@pytest.mark.parametrize(('task', 'model'), [('traj_loc_pred', 'GRU'), ('eta', 'DeepTTE')])
def test_real_dataset_and_encoder_factories(task, model):
    config = config_for(task, model)
    dataset = get_dataset(config)
    assert dataset.config is config
    assert dataset.encoder.config is config
    assert callable(dataset.get_data)


def test_eta_deeptte_forward_and_backward():
    config = config_for('eta', 'DeepTTE')
    feature = features_for('eta')
    for prefix in ('longi', 'lati', 'dist', 'dist_gap', 'time', 'time_gap'):
        feature[prefix + '_mean'] = 0.
        feature[prefix + '_std'] = 1.
    model = get_model(config, feature)
    batch = {key: torch.arange(1, 7, dtype=torch.float32).repeat(2, 1)
             for key in ('current_longi', 'current_lati', 'current_tim', 'current_dis')}
    batch['current_state'] = torch.zeros((2, 6), dtype=torch.long)
    batch.update(uid=torch.tensor([[0], [1]]), weekid=torch.tensor([[1], [2]]),
                 timeid=torch.tensor([[10], [20]]), dist=torch.tensor([[2.], [3.]]),
                 time=torch.tensor([[5.], [6.]]))
    value = model.calculate_loss(batch)
    value.backward()
    assert torch.isfinite(value)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_road_line_loss_and_backward(tmp_path):
    config = config_for('road_representation', 'LINE')
    (tmp_path / 'libcity/cache/123/evaluate_cache').mkdir(parents=True)
    model = get_model(config, features_for('road_representation'))
    value = model.calculate_loss(dict(I=torch.tensor([0, 1]), J=torch.tensor([1, 2]),
                                     Neg=torch.tensor([1, -1])))
    value.backward()
    assert torch.isfinite(value)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
