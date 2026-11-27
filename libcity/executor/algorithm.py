"""Minimal context for training backends that own their optimizer and state."""
from dataclasses import dataclass


@dataclass
class AlgorithmContext:
    model: object
    bundle: object
    config: dict
    device: object
    directory: object
    logger: object
