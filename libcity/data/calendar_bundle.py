"""Integer weekday/time-slot history from the shared, source-versioned atomic clock."""
from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

from libcity.data.traffic_bundle import prepare_traffic


def calendar_positions(clock, interval):
    if type(interval) is not int or interval<1 or 86400%interval:
        raise ValueError('Calendar adapter requires time_intervals to be a positive integer divisor of one day')
    clock = np.asarray(clock)
    if clock.dtype.kind!='M' or np.isnat(clock).any():
        raise ValueError('Calendar requires finite datetime timestamps')
    days = clock.astype('datetime64[D]')
    weekday = (days.astype(np.int64)+3)%7  # 1970-01-01 was Thursday; Monday is zero.
    slots = (clock-days)//np.timedelta64(interval,'s')
    return np.stack((weekday,slots),-1).astype(np.int64)


class CalendarWindows(Dataset):
    def __init__(self, windows, positions):
        self.windows,self.positions = windows,positions

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sample = self.windows[index]
        history = self.windows.target_starts[index]+self.windows.offsets
        sample['history_features'] = torch.from_numpy(self.positions[history].copy())
        return sample


def prepare_calendar(data,training,evaluation,seed,logger=None):
    bundle = prepare_traffic(data,training,evaluation,seed,logger)
    if bundle.metadata.input_layout['kind']!='window':
        raise ValueError('Calendar adapter requires continuous window sampling')
    interval = bundle.metadata.info.get('time_intervals',300)
    positions = calendar_positions(bundle.train.dataset.clock,interval)
    for name in ('train','val','test'):
        loader = getattr(bundle,name)
        dataset = CalendarWindows(loader.dataset,positions)
        setattr(bundle,name,DataLoader(dataset,batch_size=loader.batch_size,num_workers=loader.num_workers,
            shuffle=training['shuffle'] if name=='train' else False,generator=loader.generator))
    layout = dict(kind='calendar_positions',columns=['weekday','time_slot'],steps_per_day=86400//interval,
        time_basis='UTC',weekday_origin='Monday')
    bundle.metadata = replace(bundle.metadata,input_layout={**bundle.metadata.input_layout,'history_features':layout})
    bundle.split_summary['history_calendar'] = layout
    return bundle
