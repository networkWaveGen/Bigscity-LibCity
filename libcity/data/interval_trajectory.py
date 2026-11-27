"""Observed check-in intervals; slot bounds are fitted on training prefixes only."""
from dataclasses import replace
import json
import math

from geopy.distance import geodesic
import numpy as np
import pandas as pd
import torch

from libcity.data.task_bundles import atomic,prepare_trajectory,sessions


def observed_prefixes(data,num_nodes,*,first_time,first_distance,model_name):
    folder,raw = atomic(data)
    geo = pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    try:
        coordinates = np.asarray([json.loads(value) for value in geo['coordinates']],dtype=np.float64)
    except (ValueError,TypeError,KeyError) as error:
        raise ValueError(f'{model_name} requires JSON point coordinates in .geo') from error
    if (coordinates.shape!=(num_nodes,2) or not np.isfinite(coordinates).all()
            or (np.abs(coordinates[:,0])>180).any() or (np.abs(coordinates[:,1])>90).any()):
        raise ValueError(f'{model_name} requires finite longitude/latitude point coordinates')
    positions = {identity:point for identity,point in zip(geo['geo_id'],coordinates)}
    groups,_ = sessions(folder,raw,data)
    prefixes = {}
    for group in groups:
        if len(group)<data['min_session_length']:
            continue
        column = 'location' if 'location' in group else 'location_id'
        points = [positions[identity] for identity in group[column]]
        # Preserve the original first-check-in constants and WGS84 kilometre unit.
        distances = torch.tensor([first_distance]+[geodesic(a[::-1],b[::-1]).km
            for a,b in zip(points[:-1],points[1:])],dtype=torch.float32)
        intervals = torch.tensor(group['time'].diff().dt.total_seconds().fillna(0).to_numpy(),dtype=torch.float32)
        intervals[0] = first_time
        for position in range(1,len(group)):
            prefixes[int(group['dyna_id'].iloc[position])] = (intervals[:position],distances[:position])
    return prefixes,coordinates,groups


def prepare_intervals(data,training,evaluation,seed,logger=None):
    result = prepare_trajectory(data,training,evaluation,seed,logger)
    prefixes,_,_ = observed_prefixes(data,result.metadata.num_nodes,first_time=0.,first_distance=.5,model_name='HSTLSTM')
    for loader in (result.train,result.val,result.test):
        for row in loader.dataset.rows:
            row.pop('current_tim')
            row['tim_interval'],row['dis'] = prefixes[int(row['sample_id'])]
    # Keep the legacy lower bounds, but never scan validation/test observations.
    tim_max = math.ceil(max(100.,max(float(row['tim_interval'].max()) for row in result.train.dataset.rows)))
    dis_max = math.ceil(max(.5,max(float(row['dis'].max()) for row in result.train.dataset.rows)))
    metadata = result.metadata
    result.metadata = replace(metadata,info={**metadata.info,'tim_slot_max':tim_max,'dis_slot_max':dis_max},
        input_layout={**metadata.input_layout,'time_encoding':'observed_inter_checkin_interval',
            'time_unit':'seconds','distance_unit':'WGS84_geodesic_kilometres',
            'first_time_interval':0.,'first_distance':.5,'slot_bounds_fitted_on':'training_prefixes'})
    result.split_summary.update(slot_bounds_fitted_on='training_prefixes',tim_slot_max=tim_max,dis_slot_max=dis_max)
    return result
