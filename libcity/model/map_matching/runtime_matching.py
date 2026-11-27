"""Run each classical matcher with independent per-trajectory algorithm state."""
from importlib import import_module


class IndependentTrajectories:
    def __init__(self, name, parameters, metadata):
        self.algorithm_type = getattr(import_module('libcity.model.map_matching.'+name), name)
        self.parameters = parameters
        self.features = dict(metadata.info['features'])
        self._logger = None

    def run(self, data):
        result = {}
        for user, trajectories in data['trajectory'].items():
            result[user] = {}
            for identity, observations in trajectories.items():
                algorithm = self.algorithm_type(dict(self.parameters), dict(self.features))
                if self._logger is not None:
                    algorithm._logger = self._logger
                single = {**data, 'trajectory': {user: {identity: observations}}}
                result[user][identity] = algorithm.run(single)[user][identity]
        return result


def validate(k, r, sigma, window_size):
    if k < 1 or r <= 0 or sigma <= 0 or window_size < 1:
        raise ValueError('Map matching requires positive k, r, sigma and window_size')


def create_stmatching(*, metadata, k=5, r=100., mu=0., sigma=20., window_size=40):
    validate(k, r, sigma, window_size)
    return IndependentTrajectories('STMatching', dict(k=k, r=r, mu=mu, sigma=sigma, window_size=window_size), metadata)


def create_ivmm(*, metadata, k=5, r=100., mu=0., sigma=10., beta=40., window_size=40):
    validate(k, r, sigma, window_size)
    if beta <= 0:
        raise ValueError('IVMM beta must be positive')
    return IndependentTrajectories('IVMM', dict(k=k, r=r, mu=mu, sigma=sigma, beta=beta, window_size=window_size), metadata)
