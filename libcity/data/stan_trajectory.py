"""Hour-of-week prefixes and static spherical geometry, without dense POI pairs."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch

from libcity.data.task_bundles import atomic,prepare_personalized_trajectory,sessions


def spherical_distance(left,right):
    left,right = np.deg2rad(left),np.deg2rad(right)
    difference = left-right
    a = np.sin(difference[...,1]/2)**2+np.cos(left[...,1])*np.cos(right[...,1])*np.sin(difference[...,0]/2)**2
    a = np.clip(a,0,1)
    return 2*6371*np.arctan2(np.sqrt(a),np.sqrt(1-a))


def spherical_diameter(coordinates):
    # On the unit sphere, the nearest neighbour to -u is farthest from u:
    # ||v+u||^2 = 2+2u.v, whereas ||v-u||^2 = 2-2u.v.
    coordinates = np.unique(coordinates,axis=0)
    radians = np.deg2rad(coordinates)
    longitude,latitude = radians.T
    points = np.stack((np.cos(latitude)*np.cos(longitude),np.cos(latitude)*np.sin(longitude),np.sin(latitude)),axis=1)
    _,farthest = cKDTree(points).query(-points)
    diameter = float(spherical_distance(coordinates,coordinates[farthest]).max())
    if diameter<.001:
        # Antipodal Euclidean distances are almost 2 for a sub-metre cloud;
        # rounding can hide the nearest neighbour. Keep exact spherical values
        # with bounded tiles rather than an inaccurate diameter or a dense table.
        for start in range(0,len(coordinates),256):
            for other in range(start,len(coordinates),256):
                tile = spherical_distance(coordinates[start:start+256,None],coordinates[None,other:other+256])
                diameter = max(diameter,float(tile.max()))
    return diameter


def prepare_stan(data,training,evaluation,seed,logger=None):
    result = prepare_personalized_trajectory(data,training,evaluation,seed,logger)
    folder,raw = atomic(data)
    geo = pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    try:
        coordinates = np.asarray([json.loads(value) for value in geo['coordinates']],dtype=np.float64)
    except (ValueError,TypeError,KeyError) as error:
        raise ValueError('STAN requires JSON point coordinates in .geo') from error
    if (coordinates.shape!=(result.metadata.num_nodes,2) or not np.isfinite(coordinates).all()
            or (np.abs(coordinates[:,0])>180).any() or (np.abs(coordinates[:,1])>90).any()):
        raise ValueError('STAN requires finite longitude/latitude point coordinates')
    groups,_ = sessions(folder,raw,data)
    prefixes = {}
    for group in groups:
        if len(group)<data['min_session_length']:
            continue
        time = group['time']
        elapsed = torch.tensor((time-time.iloc[0]).dt.total_seconds().to_numpy()/3600,dtype=torch.float32)
        calendar = torch.tensor((time.dt.hour+24*time.dt.day_of_week+1).to_numpy(),dtype=torch.long)
        for position in range(1,len(group)):
            prefixes[int(group['dyna_id'].iloc[position])] = (elapsed[:position],calendar[:position])
    for loader in (result.train,result.val,result.test):
        for row in loader.dataset.rows:
            row['elapsed_hours'],row['current_tim'] = prefixes[int(row['sample_id'])]
            row['uid'] = row['uid']+1
    temporal_upper = max(float(row['elapsed_hours'][-1]) for row in result.train.dataset.rows) or 1.
    spatial_upper = spherical_diameter(coordinates) or 1.
    max_prefix = max(int(row['lengths']) for loader in (result.train,result.val,result.test) for row in loader.dataset.rows)
    origin = coordinates[0].copy()
    metadata = result.metadata
    result.metadata = replace(metadata,info={**metadata.info,'coordinates':coordinates-origin,'coordinate_origin':origin,
        'spatial_upper':spatial_upper,'temporal_upper':temporal_upper,'max_prefix_length':max_prefix,'tim_size':169,'tim_pad':0,
        'uid_size':metadata.info['uid_size']+1,'user_ids':[None,*metadata.info['user_ids']]},
        input_layout={**metadata.input_layout,'time_encoding':'hour_of_week_1_to_168','time_unit':'hours',
            'user_padding':0,'distance_unit':'great_circle_6371_kilometres',
            'query_time':'last_observation_or_explicit_known_offset','time_upper_fitted_on':'training_prefixes'})
    result.split_summary.update(query_time='last_observation',time_upper_fitted_on='training_prefixes',
        temporal_upper=temporal_upper,spatial_upper=spatial_upper,max_prefix_length=max_prefix,
        spatial_upper_source='static_POI_spherical_diameter')
    return result
