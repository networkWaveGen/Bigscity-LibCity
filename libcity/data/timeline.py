"""Target windows belong wholly to one time interval; history may precede it."""
import numpy as np


def boundaries(length, split, times=None):
    if 'train_end' in split or 'val_end' in split:
        if times is None or not {'train_end', 'val_end'} <= split.keys():
            raise ValueError('Explicit splitting requires train_end, val_end and timestamps')
        train, val = (int(np.searchsorted(times, np.datetime64(split[key])))
                      for key in ('train_end', 'val_end'))
    else:
        train, val = int(length * split['train']), int(length * (split['train'] + split['val']))
    if not 0 < train < val < length:
        raise ValueError('Time split produces empty train/validation/test splits')
    return train, val


def partition_windows(starts, horizon, cuts, length):
    starts = np.asarray(starts, dtype=np.int64)
    return [np.flatnonzero((starts >= left) & (starts + horizon <= right))
            for left, right in zip((0, *cuts), (*cuts, length))]
