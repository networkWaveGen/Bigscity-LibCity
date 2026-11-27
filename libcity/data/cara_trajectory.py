"""Observed calendar/time gaps and static POIs for context-aware recurrence."""
from dataclasses import replace

import torch

from libcity.data.task_bundles import atomic,prepare_personalized_trajectory,sessions
from libcity.data.trajectory_points import read_points


def prepare_cara(data,training,evaluation,seed,logger=None):
    result = prepare_personalized_trajectory(data,training,evaluation,seed,logger)
    folder,raw = atomic(data)
    _,coordinates = read_points(folder,raw,data['dataset'],result.metadata.num_nodes,'CARA')
    groups,_ = sessions(folder,raw,data)
    prefixes = {}
    for group in groups:
        if len(group)<data['min_session_length']:
            continue
        time = group['time']
        gaps = torch.tensor(time.diff().dt.total_seconds().fillna(0).to_numpy()/3600,dtype=torch.float32)
        week = torch.tensor((24*time.dt.day_of_week+time.dt.hour+time.dt.minute/60+time.dt.second/3600).to_numpy(),dtype=torch.float32)
        for position in range(1,len(group)):
            prefixes[int(group['dyna_id'].iloc[position])] = (gaps[:position],week[position-1])
    for loader in (result.train,result.val,result.test):
        for row in loader.dataset.rows:
            row['time_gaps'],row['last_week_hour'] = prefixes[int(row['sample_id'])]
    metadata = result.metadata
    origin = coordinates[0].copy()
    result.metadata = replace(metadata,info={**metadata.info,'coordinates':coordinates-origin,'coordinate_origin':origin},
        input_layout={**metadata.input_layout,'time_gap_unit':'hours','time_gap_source':'observed_timestamps',
            'distance_unit':'great_circle_6373_kilometres','query_time':'last_observation_or_explicit_known_offset'})
    result.split_summary.update(query_time='last_observation',time_gap_source='observed_timestamps',
        distance_unit='great_circle_6373_kilometres',evaluation_candidates='full_location_vocabulary')
    return result
