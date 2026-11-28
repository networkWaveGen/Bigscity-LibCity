"""Distinct, named periodic protocols; every offset is relative to target start."""
import numpy as np


def periodic_offsets(sampling,interval,input_window):
    name = sampling['name']
    if name=='seasonal_windows':
        # Preserve CONVGCN's service-period offsets (not calendar day/week units).
        offsets = np.concatenate([np.arange(-shift-input_window,-shift) for shift in
            (sampling['trend_steps'],sampling['period_steps'],0)])
        return offsets,dict(kind=name,length=len(offsets),window_length=input_window,
            segment_order=['trend','period','closeness'],sampling=dict(sampling))
    if name=='frame_cpt':
        if interval<1 or 86400%interval:
            raise ValueError('frame_cpt requires time_intervals dividing one day')
        offsets = [-index for index in range(1,sampling.get('closeness',0)+1)]
        offsets += [-index*(86400//interval)*sampling.get('interval_period',1)
                    for index in range(1,sampling.get('period',0)+1)]
        offsets += [-index*(7*86400//interval)*sampling.get('interval_trend',1)
                    for index in range(1,sampling.get('trend',0)+1)]
        return np.asarray(offsets,dtype=np.int64),dict(kind=name,length=len(offsets),
            segment_order=['closeness','period','trend'],sampling=dict(sampling))
    raise ValueError(f'Unsupported periodic sampling {name}')
