"""Exact physical-time offsets for active hourly, daily and weekly histories."""
import numpy as np


def cpt_offsets(sampling, interval, horizon):
    if type(interval) is not int or interval<1:
        raise ValueError('CPT time_intervals must be a positive integer number of seconds')
    if type(horizon) is not int or horizon<1:
        raise ValueError('CPT output_window must be a positive integer')
    offsets = []
    for name,seconds,stride in (
            ('trend',7*86400,sampling.get('interval_trend',1)),
            ('period',86400,sampling.get('interval_period',1)),
            ('closeness',3600,1)):
        count = sampling.get(name,0)
        if type(count) is not int or count<0:
            raise ValueError(f'CPT {name} count must be a nonnegative integer')
        if not count:
            continue
        if type(stride) is not int or stride<1 or seconds*stride%interval:
            raise ValueError(f'CPT {name} shift must be exactly representable by time_intervals')
        shift = seconds*stride//interval
        if horizon>shift:
            raise ValueError(f'CPT output_window exceeds the {name} history shift; inputs would reach the target window')
        offsets.extend(-depend*shift+position for depend in range(1,count+1) for position in range(horizon))
    if not offsets:
        raise ValueError('CPT requires at least one active history segment')
    return np.asarray(offsets,dtype=np.int64)
