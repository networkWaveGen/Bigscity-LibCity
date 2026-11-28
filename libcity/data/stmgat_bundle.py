"""Continuous point windows retain raw targets while imputing only history."""
from libcity.data.traffic_bundle import prepare_finite_traffic


def prepare_stmgat(data,training,evaluation,seed,logger=None):
    if data['sampling']['name']!='window': raise ValueError('STMGAT requires continuous windows')
    return prepare_finite_traffic(data,training,evaluation,seed,logger)
