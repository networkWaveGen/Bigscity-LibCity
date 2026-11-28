"""Continuous point windows retain raw targets while imputing only history."""
from dataclasses import replace

import numpy as np

from libcity.data.traffic_bundle import prepare_traffic


def prepare_stmgat(data,training,evaluation,seed,logger=None):
    if data['sampling']['name']!='window': raise ValueError('STMGAT requires continuous windows')
    bundle=prepare_traffic(data,training,evaluation,seed,logger)
    history=bundle.train.dataset.history
    if np.isinf(history).any(): raise ValueError('STMGAT historical observations must not contain infinities')
    cleaned=np.nan_to_num(history,nan=0.)
    for loader in (bundle.train,bundle.val,bundle.test): loader.dataset.history=cleaned
    bundle.metadata=replace(bundle.metadata,input_layout={**bundle.metadata.input_layout,
        'input_missing_observations':'raw_zero_before_scaling'})
    bundle.split_summary['input_missing_observations']='raw_zero_before_scaling'
    return bundle
