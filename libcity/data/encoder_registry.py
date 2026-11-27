"""Lazy encoder lookup for retained, pending dataset source; no experiment factories."""
from importlib import import_module

ENCODERS = {'traj_encoder': {'AtstlstmEncoder': 'data.dataset.trajectory_encoder.atstlstm_encoder', 'CARATrajectoryEncoder': 'data.dataset.trajectory_encoder.cara_encoder', 'HstlstmEncoder': 'data.dataset.trajectory_encoder.hstlstm_encoder', 'LstpmEncoder': 'data.dataset.trajectory_encoder.lstpm_encoder', 'SermEncoder': 'data.dataset.trajectory_encoder.serm_encoder', 'StanEncoder': 'data.dataset.trajectory_encoder.stan_encoder', 'StandardTrajectoryEncoder': 'data.dataset.trajectory_encoder.standard_trajectory_encoder', 'StrnnEncoder': 'data.dataset.trajectory_encoder.strnn_encoder'}, 'eta_encoder': {'DeeptteEncoder': 'data.dataset.eta_encoder.deeptte_encoder', 'TtpnetEncoder': 'data.dataset.eta_encoder.ttpnet_encoder'}}


def load_encoder(kind,name):
    try:
        module=ENCODERS[kind][name]
    except KeyError:
        raise ValueError(f'Unknown encoder {kind}/{name}') from None
    return getattr(import_module('libcity.'+module),name)
