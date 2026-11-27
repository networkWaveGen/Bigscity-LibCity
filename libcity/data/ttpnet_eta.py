"""Departure-known speed profiles and static road embeddings for TTPNet."""
import json

import numpy as np
import pandas as pd
import torch

from libcity.data.bundle import DatasetMetadata
from libcity.data.eta_routes import route_records
from libcity.data.task_bundles import bundle,safe_statistics
from libcity.utils.normalization import StandardScaler


SPEED_COLUMNS = ('speeds','speeds_relevant1','speeds_relevant2','speeds_long')


def json_vectors(values,width,name):
    try:
        result = np.asarray([json.loads(value) for value in values],dtype=np.float64)
    except (ValueError,TypeError) as error:
        raise ValueError(f'TTPNet {name} must contain JSON numeric vectors') from error
    if result.ndim!=2 or (width is not None and result.shape[1]!=width) or not np.isfinite(result).all():
        raise ValueError(f'TTPNet {name} requires finite vectors'+(f' of width {width}' if width else ''))
    return result


def prepare_ttpnet(data,training,evaluation,seed,logger=None):
    folder,raw,users,trips,cuts = route_records(data,minimum=data['min_route_length'])
    geo = pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    if 'geo_id' not in geo or 'embeddings' not in geo or geo.empty or geo.geo_id.isna().any() or geo.geo_id.duplicated().any():
        raise ValueError('TTPNet requires distinct .geo IDs with JSON embeddings')
    embeddings = json_vectors(geo['embeddings'],None,'road embeddings')
    if embeddings.shape[1]<1:
        raise ValueError('TTPNet road embeddings must have positive width')
    vocabulary = {identity:index for index,identity in enumerate(geo.geo_id)}
    speed_unit,grid_unit = raw.get('speed_unit','km/h'),raw.get('grid_len_unit','km')
    if speed_unit not in ('km/h','m/s') or grid_unit not in ('km','m'):
        raise ValueError('TTPNet supports speed_unit km/h or m/s and grid_len_unit km or m')
    snapshot_column = raw.get('traffic_snapshot_time_column','traffic_snapshot_time')
    if not isinstance(snapshot_column,str) or not snapshot_column:
        raise ValueError('TTPNet traffic_snapshot_time_column must be a nonempty column name')
    for part in trips:
        for trip in part:
            group = trip['group']
            if snapshot_column not in group:
                raise ValueError('TTPNet requires traffic_snapshot_time (or declared traffic_snapshot_time_column) at/before departure')
            snapshot = pd.to_datetime(group[snapshot_column],utc=True,errors='coerce',format='mixed')
            if snapshot.isna().any() or (snapshot>group['time'].iloc[0]).any():
                raise ValueError('TTPNet traffic snapshot must be finite and available at/before departure')
            if 'grid_len' not in group or any(name not in group for name in SPEED_COLUMNS):
                raise ValueError('TTPNet requires grid_len and all four speed-profile columns')
            for name in SPEED_COLUMNS:
                trip[name] = json_vectors(group[name],7 if name=='speeds_long' else 4,name)*(3.6 if speed_unit=='m/s' else 1.)
                if (trip[name]<0).any():
                    raise ValueError('TTPNet speed profiles must be nonnegative')
            trip['grid_len'] = group['grid_len'].to_numpy(dtype=np.float64)*(.001 if grid_unit=='m' else 1.)
            if not np.isfinite(trip['grid_len']).all() or (trip['grid_len']<0).any():
                raise ValueError('TTPNet grid_len must be finite and nonnegative')
            column = 'location' if 'location' in group else 'location_id'
            if column not in group:
                raise ValueError('TTPNet route must provide location IDs for the static .geo embeddings')
            try:
                trip['road_id'] = np.asarray([vocabulary[identity] for identity in group[column]],dtype=np.int64)
            except KeyError as error:
                raise ValueError('TTPNet route location is absent from .geo') from error
            holiday = group['holiday'].iloc[0] if 'holiday' in group else int(trip['weekid']>=5)
            if holiday not in (0,1):
                raise ValueError('TTPNet departure holiday must be 0 or 1')
            trip['holiday'] = holiday
    train = trips[0]
    coords = np.concatenate([trip['coords'] for trip in train])
    stats = dict(longi=safe_statistics(coords[:,0]),lati=safe_statistics(coords[:,1]),
        dist=safe_statistics([trip['distances'][-1] for trip in train]),
        time=safe_statistics([trip['durations'][-1] for trip in train]),
        time_gap=safe_statistics(np.concatenate([np.diff(trip['durations']) for trip in train])))
    for name in (*SPEED_COLUMNS,'grid_len'):
        stats[name] = safe_statistics(np.concatenate([trip[name].reshape(-1) for trip in train]))
    scaler = StandardScaler(*stats['time_gap'])
    def normalized(values,name):
        mean,std = stats[name]
        return (values-mean)/std
    rows = [[],[],[]]
    for index,part in enumerate(trips):
        for trip in part:
            features = []
            for name in SPEED_COLUMNS:
                speeds = trip[name]
                distance = np.broadcast_to(trip['grid_len'][:,None],speeds.shape)
                travel = distance/np.where(speeds==0,.2,speeds)*3600
                features.append(np.stack((normalized(speeds,name),normalized(distance,'grid_len'),normalized(travel,'time_gap')),axis=-1))
            row = dict(coords=torch.tensor(np.stack((normalized(trip['coords'][:,0],'longi'),normalized(trip['coords'][:,1],'lati')),-1),dtype=torch.float32),
                road_id=torch.tensor(trip['road_id']),short_traffic=torch.tensor(np.stack(features[:3],axis=1),dtype=torch.float32),
                long_traffic=torch.tensor(features[3],dtype=torch.float32),dist=torch.tensor([normalized(trip['distances'][-1],'dist')],dtype=torch.float32),
                holiday=torch.tensor([trip['holiday']],dtype=torch.float32),lengths=torch.tensor(len(trip['coords'])),
                local_y_raw=torch.tensor(trip['durations'],dtype=torch.float32),
                y=torch.tensor([scaler.transform(trip['durations'][-1])],dtype=torch.float32),
                y_raw=torch.tensor([trip['durations'][-1]],dtype=torch.float32),
                timeid=torch.tensor(trip['timeid']//15))
            row.update({name:torch.tensor(trip[name]) for name in ('uid','weekid','traj_id','sample_id')})
            rows[index].append(row)
    metadata = DatasetMetadata(data['dataset'],output_dim=1,info=dict(uid_size=len(users),user_ids=users,
        road_ids=geo.geo_id.tolist(),road_embeddings=embeddings.astype(np.float32),statistics=stats),
        input_layout=dict(kind='planned_route',target_semantics='duration_seconds',coordinates_normalized=True,
            traffic_inputs='departure_known_snapshot',traffic_snapshot_time_column=snapshot_column,
            short_traffic_layout='point_branch_4slots_speed_length_time',long_traffic_layout='point_7slots_speed_length_time',
            speed_unit='km/h',distance_unit='km',prediction_scale='training_time_gap',
            time_encoding='departure_15_minute_slot',minimum_route_length=data['min_route_length']))
    return bundle(rows,metadata,scaler,training,evaluation,seed,dict(semantics='complete_trip_time',duration_unit='seconds',
        train_end=str(cuts[0]),val_end=str(cuts[1]),traffic_inputs='departure_known_snapshot',
        minimum_route_length=data['min_route_length'],input_speed_unit=speed_unit,input_grid_len_unit=grid_unit,
        speed_unit='km/h',distance_unit='km',prediction_scale='training_time_gap'))
