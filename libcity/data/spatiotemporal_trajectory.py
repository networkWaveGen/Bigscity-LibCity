"""Known prefix times and static POI geometry for candidate-aware trajectory scores."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import torch

from libcity.data.task_bundles import atomic,prepare_personalized_trajectory,sessions


def prepare_spatiotemporal(data,training,evaluation,seed,logger=None):
    result = prepare_personalized_trajectory(data,training,evaluation,seed,logger)
    folder,raw = atomic(data)
    geo = pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    try:
        coordinates = np.asarray([json.loads(value) for value in geo['coordinates']],dtype=np.float64)
    except (ValueError,TypeError,KeyError) as error:
        raise ValueError('STRNN requires JSON point coordinates in .geo') from error
    if coordinates.shape!=(result.metadata.num_nodes,2) or not np.isfinite(coordinates).all():
        raise ValueError('STRNN requires finite two-dimensional point coordinates')
    origin = coordinates[0].copy()
    coordinates = coordinates-origin
    distance_upper = float(np.linalg.norm(np.ptp(coordinates,axis=0))) or 1.
    groups,_ = sessions(folder,raw,data)
    prefixes = {}
    for group in groups:
        if len(group)<data['min_session_length']:
            continue
        # Retain sub-hour information; the actual next-event time is never input.
        elapsed = torch.tensor((group['time']-group['time'].iloc[0]).dt.total_seconds().to_numpy()/3600,dtype=torch.float32)
        for position in range(1,len(group)):
            prefixes[int(group['dyna_id'].iloc[position])] = elapsed[:position]
    for loader in (result.train,result.val,result.test):
        for row in loader.dataset.rows:
            row['elapsed_hours'] = prefixes[int(row['sample_id'])]
    time_upper = max(float(row['elapsed_hours'][-1]-row['elapsed_hours'][0]) for row in result.train.dataset.rows) or 1.
    metadata = result.metadata
    result.metadata = replace(metadata,info={**metadata.info,'coordinates':coordinates,'coordinate_origin':origin,
        'distance_upper':distance_upper,'time_upper':time_upper},input_layout={**metadata.input_layout,
        'query_time':'last_observation_or_explicit_known_offset','time_unit':'hours',
        'distance_unit':'euclidean_coordinate_degrees','distance_features':'per_candidate'})
    result.split_summary.update(query_time='last_observation',time_upper_fitted_on='training_prefixes',
        time_upper=time_upper,distance_upper=distance_upper)
    return result
