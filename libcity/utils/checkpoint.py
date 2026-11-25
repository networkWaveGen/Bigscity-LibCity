"""Read and restore historical state-dict checkpoint formats explicitly."""

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


def load_checkpoint(path, device='cpu', *, trusted_legacy=False):
    """Load weights safely; object-pickle migration requires explicit opt-in."""
    return torch.load(path, map_location=device, weights_only=not trusted_legacy)


def unpack_checkpoint(checkpoint):
    """Return model, optional optimizer and epoch from supported old formats."""
    if isinstance(checkpoint, (tuple, list)) and len(checkpoint) == 2:
        model, optimizer = checkpoint
        epoch = None
    elif isinstance(checkpoint, Mapping) and 'model_state_dict' in checkpoint:
        model = checkpoint['model_state_dict']
        optimizer = checkpoint.get('optimizer_state_dict')
        epoch = checkpoint.get('epoch')
    elif isinstance(checkpoint, Mapping) and 'state_dict' in checkpoint:
        model = checkpoint['state_dict']
        optimizer = checkpoint.get('optimizer_state_dict')
        epoch = checkpoint.get('epoch')
    elif isinstance(checkpoint, Mapping):
        model, optimizer, epoch = checkpoint, None, None
    else:
        raise ValueError('Unsupported checkpoint format; expected state dictionaries, '
                         'a (model, optimizer) pair, or an epoch dictionary.')
    if not isinstance(model, Mapping) or not all(isinstance(key, str) for key in model):
        raise ValueError('Checkpoint model state must be a mapping with string keys.')
    if optimizer is not None and not isinstance(optimizer, Mapping):
        raise ValueError('Checkpoint optimizer state must be a mapping.')
    return model, optimizer, epoch


def restore_checkpoint(path, model, optimizer=None, device='cpu'):
    """Restore model and, when requested, a real optimizer state; return epoch."""
    state, optimizer_state, epoch = unpack_checkpoint(load_checkpoint(path, device))
    if optimizer is not None and optimizer_state is None:
        raise ValueError('This checkpoint contains only model weights. Resume training '
                         'requires an optimizer state; load weights with optimizer=None.')
    model.load_state_dict(state)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    return epoch


def migrate_checkpoint(source, destination, *, trusted_legacy=False):
    """Convert trusted object pickles or old dictionaries to CPU state dicts.

    Ordinary tensor checkpoints need no unsafe opt-in. Never automatically retry
    a failed weights-only read with object unpickling enabled.
    """
    checkpoint = load_checkpoint(source, 'cpu', trusted_legacy=trusted_legacy)
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, Mapping) and isinstance(checkpoint.get('model'), torch.nn.Module):
        optimizer = checkpoint.get('optimizer')
        checkpoint = dict(model_state_dict=checkpoint['model'].state_dict(),
                          optimizer_state_dict=optimizer.state_dict() if optimizer is not None else None,
                          epoch=checkpoint.get('epoch'))
    model, optimizer, epoch = unpack_checkpoint(checkpoint)
    if epoch is not None:
        normalized = dict(model_state_dict=model, optimizer_state_dict=optimizer, epoch=epoch)
    elif optimizer is not None:
        normalized = (model, optimizer)
    else:
        normalized = model
    torch.save(normalized, destination)
