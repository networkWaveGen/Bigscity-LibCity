"""Complete-trip ETA records shared by feature adapters, before normalization."""
import json

import numpy as np
import pandas as pd

from libcity.data.task_bundles import atomic,sessions,split_index
from libcity.data.trajectory_points import spherical_distance


def route_records(data,*,minimum=3):
    folder,raw = atomic(data)
    groups,cuts = sessions(folder,raw,data)
    users = pd.read_csv(folder/(raw.get('usr_file',data['dataset'])+'.usr'))['usr_id'].tolist()
    if not users or pd.isna(users).any() or len(set(users))!=len(users):
        raise ValueError('ETA user vocabulary must be nonempty and unique')
    vocabulary = {user:index for index,user in enumerate(users)}
    trips = [[],[],[]]
    distance_unit = raw.get('distance_unit','km')
    if distance_unit not in ('km','m'):
        raise ValueError('ETA distance_unit must be km or m')
    for group in groups:
        if len(group)<minimum:
            continue
        time = group['time']
        if time.isna().any():
            raise ValueError('ETA timestamps must be finite')
        part = split_index(time.iloc[0],time.iloc[-1],cuts)
        if part is None:
            continue
        try:
            coords = np.asarray([json.loads(item) for item in group['coordinates']],dtype=np.float64)
        except (ValueError,TypeError,KeyError) as error:
            raise ValueError('ETA requires JSON planned-route coordinates') from error
        if (coords.shape!=(len(group),2) or not np.isfinite(coords).all()
                or (np.abs(coords[:,0])>180).any() or (np.abs(coords[:,1])>90).any()):
            raise ValueError('ETA coordinates must be finite longitude/latitude pairs')
        distances = (group['current_dis'].to_numpy(dtype=np.float64)*(1. if distance_unit=='km' else .001)
            if 'current_dis' in group else np.r_[0.,np.cumsum(spherical_distance(coords[:-1],coords[1:]))])
        durations = (time-time.iloc[0]).dt.total_seconds().to_numpy()
        if (not np.isfinite(distances).all() or not np.isfinite(durations).all()
                or np.any(np.diff(durations)<=0) or np.any(np.diff(distances)<0)):
            raise ValueError('ETA trip time and cumulative distance must increase and be finite')
        user = group['entity_id'].iloc[0]
        if user not in vocabulary:
            raise ValueError('ETA user is absent from the .usr vocabulary')
        trips[part].append(dict(coords=coords,distances=distances-distances[0],durations=durations,
            uid=vocabulary[user],weekid=time.iloc[0].day_of_week,
            timeid=time.iloc[0].hour*60+time.iloc[0].minute,traj_id=int(group['traj_id'].iloc[0]),
            sample_id=int(group['dyna_id'].iloc[0]),group=group))
    if any(not part for part in trips):
        raise ValueError('ETA needs nonempty disjoint complete-trip intervals')
    return folder,raw,users,trips,cuts
