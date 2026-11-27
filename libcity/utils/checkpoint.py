"""Own CPU snapshots of model and optimizer tensor state."""

from collections.abc import Mapping
from copy import deepcopy
import torch


def copy_state_to_cpu(state):
    """Own a CPU snapshot while retaining state-dict metadata and containers."""
    memo = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            memo[id(value)] = value.detach().to('cpu', copy=True)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(state)
    return deepcopy(state, memo)
