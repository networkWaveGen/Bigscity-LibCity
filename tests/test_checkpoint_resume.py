"""Exercise nonempty Adam state through actual executor save/load entrypoints."""

import copy
import logging
import pickle

import pytest
import torch

from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.executor.traj_loc_pred_executor import TrajLocPredExecutor
from libcity.utils.checkpoint import (
    load_checkpoint, migrate_checkpoint, restore_checkpoint, unpack_checkpoint,
)


def step(model, optimizer):
    optimizer.zero_grad()
    value = (model(torch.tensor([[1., 2.], [3., 4.]])) - 2).square().mean()
    value.backward()
    optimizer.step()
    return value.detach()


def bare_executor(kind, directory):
    cls = TrajLocPredExecutor if kind == 'trajectory' else TrafficStateExecutor
    executor = object.__new__(cls)
    executor.model = torch.nn.Linear(2, 1)
    executor.optimizer = torch.optim.Adam(executor.model.parameters(), lr=0.03)
    executor.device = torch.device('cpu')
    executor.cache_dir = str(directory)
    executor.config = dict(device=executor.device, model='Linear', dataset='fixture')
    executor._logger = logging.getLogger(__name__)
    return executor


@pytest.mark.parametrize('kind', ['traffic', 'trajectory', 'epoch'])
def test_adam_resume_matches_uninterrupted_next_step(tmp_path, kind):
    torch.manual_seed(17)
    original = bare_executor(kind, tmp_path)
    step(original.model, original.optimizer)
    saved_state = copy.deepcopy(original.optimizer.state_dict())
    assert saved_state['state']  # Adam moments exist only after a real step.
    if kind == 'epoch':
        original.save_model_with_epoch(1)
    else:
        original.save_model(str(tmp_path / 'pair.m'))
    expected_loss = step(original.model, original.optimizer)
    resumed = bare_executor(kind, tmp_path)
    if kind == 'epoch':
        resumed.load_model_with_epoch(1)
    else:
        resumed.load_model(str(tmp_path / 'pair.m'))
    for entry in resumed.optimizer.state.values():
        assert entry['step'].item() == 1
        assert entry['exp_avg'].device.type == 'cpu'
    actual_loss = step(resumed.model, resumed.optimizer)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for actual, expected in zip(resumed.model.parameters(), original.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(resumed.optimizer.state.values(), original.optimizer.state.values()):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_device_forwarding_and_weights_only_default(monkeypatch):
    arguments = {}
    monkeypatch.setattr(torch, 'load', lambda path, **kwargs: arguments.update(kwargs))
    load_checkpoint('fixture', torch.device('cuda:3'))
    assert arguments == dict(map_location=torch.device('cuda:3'), weights_only=True)


def test_model_only_checkpoint_cannot_silently_reset_optimizer(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / 'weights.pt'
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match='only model weights'):
        restore_checkpoint(path, model, torch.optim.Adam(model.parameters()))
    restore_checkpoint(path, model)


@pytest.mark.parametrize('format_name', ['pair', 'epoch', 'state_dict', 'weights', 'object'])
def test_checkpoint_migration_keeps_values_and_optimizer(tmp_path, format_name):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    step(model, optimizer)
    values = dict(pair=(model.state_dict(), optimizer.state_dict()),
                  epoch=dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), epoch=4),
                  state_dict=dict(state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), epoch=4),
                  weights=model.state_dict(), object=dict(model=model, optimizer=optimizer, epoch=4))
    source, target = tmp_path / 'old.pt', tmp_path / 'new.pt'
    torch.save(values[format_name], source)
    if format_name == 'object':
        with pytest.raises(pickle.UnpicklingError):
            migrate_checkpoint(source, target)
    migrate_checkpoint(source, target, trusted_legacy=format_name == 'object')
    state, opt, epoch = unpack_checkpoint(load_checkpoint(target))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(state[key], value)
    assert (opt is None) == (format_name == 'weights')
    if opt is not None:
        assert all(value['step'].item() == 1 for value in opt['state'].values())
    assert epoch == (4 if format_name in ('epoch', 'state_dict', 'object') else None)
