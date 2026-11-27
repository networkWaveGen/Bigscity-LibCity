"""Small atomic adapters for variable-length trajectories, ETA and road edges."""
import json
from bisect import bisect_left
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from libcity.data.bundle import DataBundle, DatasetMetadata
from libcity.data.timeline import boundaries
from libcity.utils.normalization import NoneScaler, StandardScaler


def atomic(data):
    folder=Path(data['data_dir'])/data['dataset']
    raw=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    return folder,{**raw,**raw.get('info',{})}


class Records(Dataset):
    def __init__(self,rows):
        self.rows=rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        return self.rows[index]


def collate_sequences(rows):
    result={}
    for key in rows[0]:
        values=[row[key] for row in rows]
        result[key]=pad_sequence(values,batch_first=True) if values[0].ndim else torch.stack(values)
    return result


def bundle(rows,metadata,scaler,training,evaluation,seed,summary):
    generator=torch.Generator().manual_seed(seed)
    loaders=[]
    for index,part in enumerate(rows):
        if not part:
            raise ValueError('Atomic task has empty train/validation/test split')
        cfg=training if index==0 else evaluation
        loaders.append(DataLoader(Records(part),batch_size=cfg['batch_size'],num_workers=cfg['num_workers'],
            shuffle=training['shuffle'] if index==0 else False,generator=generator if index==0 else None,
            collate_fn=collate_sequences))
    return DataBundle(*loaders,metadata,scaler,generator,
        dict(train=len(rows[0]),val=len(rows[1]),test=len(rows[2]),**summary))


def sessions(folder,raw,data):
    name=raw.get('dyna_file',data['dataset'])
    frame=pd.read_csv(folder/(name+'.dyna'))
    frame['time']=pd.to_datetime(frame['time'],utc=True)
    frame=frame.sort_values(['entity_id','time','dyna_id'])
    if 'traj_id' not in frame:
        gap=frame.groupby('entity_id')['time'].diff().dt.total_seconds()
        frame['traj_id']=(gap.isna() | (gap>data.get('session_gap_hours',24.)*3600)).cumsum()
    groups=[part for _,part in frame.groupby(['entity_id','traj_id'],sort=False)]
    times=np.sort(frame['time'].dt.tz_localize(None).unique().astype('datetime64[ns]'))
    cuts=boundaries(len(times),data['split'],times)
    clock=(times[cuts[0]],times[cuts[1]])
    return groups,clock


def split_index(start,end,cuts):
    start,end=np.datetime64(start.tz_localize(None)),np.datetime64(end.tz_localize(None))
    if end<cuts[0]:
        return 0
    if start>=cuts[0] and end<cuts[1]:
        return 1
    if start>=cuts[1]:
        return 2
    return None


def history_banks(encoded_groups):
    users = {}
    for group,locations,times in encoded_groups:
        users.setdefault(group['entity_id'].iloc[0],[]).append((group['time'].iloc[-1].value,locations,times))
    result = {}
    for user,entries in users.items():
        entries.sort(key=lambda row:row[0])
        endings,offsets,locations,times = [],[0],[],[]
        for ending,loc,tim in entries:
            endings.append(ending)
            locations.extend(loc);times.extend(tim)
            offsets.append(len(locations))
        result[user] = (endings,offsets,torch.tensor(locations,dtype=torch.long),torch.tensor(times,dtype=torch.long))
    return result


def prepare_trajectory(data,training,evaluation,seed,logger=None,*,include_users=False,include_history=False):
    folder,raw=atomic(data)
    geo=pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    ids=geo['geo_id'].tolist()
    if not ids or len(set(ids))!=len(ids):
        raise ValueError('Trajectory location vocabulary must be nonempty and unique')
    locations={value:index+1 for index,value in enumerate(ids)}
    groups,cuts=sessions(folder,raw,data)
    users = None
    if include_users:
        path = folder/(raw.get('usr_file',data['dataset'])+'.usr')
        identities = pd.read_csv(path)['usr_id'].tolist() if path.is_file() else list(dict.fromkeys(
            group['entity_id'].iloc[0] for group in groups))
        if not identities or len(set(identities)) != len(identities):
            raise ValueError('Trajectory user vocabulary must be nonempty and unique')
        users = {identity:index for index,identity in enumerate(identities)}
    encoded_groups = []
    for group in groups:
        if len(group)<data['min_session_length']:
            continue
        location_column='location' if 'location' in group else 'location_id'
        try:
            loc=[locations[item] for item in group[location_column]]
        except KeyError as error:
            raise ValueError('Trajectory location is absent from .geo vocabulary') from error
        time=group['time']
        if time.isna().any():
            raise ValueError('Trajectory timestamps must be finite')
        encoded=(time.dt.hour+24*(time.dt.day_of_week>=5)).tolist()
        encoded_groups.append((group,loc,encoded))
    histories = history_banks(encoded_groups) if include_history else None
    rows=[[],[],[]]
    for group,loc,encoded in encoded_groups:
        time = group['time']
        history = {}
        if histories is not None:
            endings,offsets,bank_loc,bank_tim = histories[group['entity_id'].iloc[0]]
            stop = offsets[bisect_left(endings,time.iloc[0].value)]
            # Views share the immutable user bank; collation creates owned tensors.
            history = dict(history_loc=bank_loc[:stop],history_tim=bank_tim[:stop],history_lengths=torch.tensor(stop))
        for position in range(1,len(group)):
            part=split_index(time.iloc[position],time.iloc[position],cuts)
            row = dict(current_loc=torch.tensor(loc[:position]),current_tim=torch.tensor(encoded[:position]),
                lengths=torch.tensor(position),y=torch.tensor(loc[position]-1),
                sample_id=torch.tensor(int(group['dyna_id'].iloc[position])))
            if users is not None:
                user = group['entity_id'].iloc[0]
                if user not in users:
                    raise ValueError('Trajectory user is absent from .usr vocabulary')
                row['uid'] = torch.tensor(users[user])
            row.update(history)
            rows[part].append(row)
    info = dict(location_ids=ids,loc_size=len(ids)+1,tim_size=49,loc_pad=0,tim_pad=48)
    if users is not None:
        info.update(uid_size=len(users),user_ids=identities)
    layout = dict(kind='padded_trajectory',target_semantics='next_location',time_encoding='weekday_weekend_hour')
    summary = dict(semantics='target_timestamp',train_end=str(cuts[0]),val_end=str(cuts[1]))
    if include_history:
        layout.update(history_semantics='completed_before_session_start',history_order='completion_time',empty_history='zero_context')
        summary['cold_start_samples'] = {name:sum(int(row['history_lengths'])==0 for row in part)
            for name,part in zip(('train','val','test'),rows)}
    metadata=DatasetMetadata(data['dataset'],num_nodes=len(ids),output_dim=len(ids),info=info,input_layout=layout)
    return bundle(rows,metadata,NoneScaler(),training,evaluation,seed,
        summary)


def prepare_personalized_trajectory(data,training,evaluation,seed,logger=None):
    return prepare_trajectory(data,training,evaluation,seed,logger,include_users=True)


def prepare_history_trajectory(data,training,evaluation,seed,logger=None):
    return prepare_trajectory(data,training,evaluation,seed,logger,include_history=True)


def safe_statistics(values):
    values=np.asarray(values,dtype=np.float64)
    mean,std=float(values.mean()),float(values.std())
    return mean,std if std>1e-8 else 1.


def prepare_eta(data,training,evaluation,seed,logger=None):
    from libcity.data.dataset.eta_encoder.deeptte_encoder import geo_distance
    folder,raw=atomic(data)
    groups,cuts=sessions(folder,raw,data)
    users=pd.read_csv(folder/(raw.get('usr_file',data['dataset'])+'.usr'))['usr_id'].tolist()
    vocabulary={user:index for index,user in enumerate(users)}
    trips=[[],[],[]]
    for group in groups:
        if len(group)<3:
            continue
        time=group['time']
        part=split_index(time.iloc[0],time.iloc[-1],cuts)
        if part is None:
            continue
        coords=np.array([json.loads(item) for item in group['coordinates']])
        if coords.shape!=(len(group),2) or not np.isfinite(coords).all():
            raise ValueError('ETA coordinates must be finite longitude/latitude pairs')
        distances=group['current_dis'].to_numpy() if 'current_dis' in group else np.r_[0.,np.cumsum([
            geo_distance(*a,*b) for a,b in zip(coords[:-1],coords[1:])])]
        durations=(time-time.iloc[0]).dt.total_seconds().to_numpy()
        if np.any(np.diff(durations)<=0) or np.any(np.diff(distances)<0):
            raise ValueError('ETA trip time and cumulative distance must increase')
        trips[part].append(dict(coords=coords,distances=distances-distances[0],durations=durations,
            state=group['current_state'].to_numpy() if 'current_state' in group else np.zeros(len(group),dtype=int),
            uid=vocabulary[group['entity_id'].iloc[0]],weekid=time.iloc[0].day_of_week,
            timeid=time.iloc[0].hour*60+time.iloc[0].minute,traj_id=int(group['traj_id'].iloc[0]),
            sample_id=int(group['dyna_id'].iloc[0])))
    if any(not part for part in trips):
        raise ValueError('ETA needs nonempty disjoint complete-trip intervals')
    train=trips[0]
    coordinates=np.concatenate([trip['coords'] for trip in train])
    stats=dict(longi=safe_statistics(coordinates[:,0]),lati=safe_statistics(coordinates[:,1]),
        dist=safe_statistics([trip['distances'][-1] for trip in train]),
        time=safe_statistics([trip['durations'][-1] for trip in train]),
        dist_gap=safe_statistics(np.concatenate([np.diff(trip['distances']) for trip in train])),
        time_gap=safe_statistics(np.concatenate([np.diff(trip['durations']) for trip in train])))
    scaler=StandardScaler(*stats['time'])
    rows=[[],[],[]]
    for index,part in enumerate(trips):
        for trip in part:
            coords=trip['coords'].copy()
            coords[:,0]=(coords[:,0]-stats['longi'][0])/stats['longi'][1]
            coords[:,1]=(coords[:,1]-stats['lati'][0])/stats['lati'][1]
            row=dict(coords=torch.tensor(coords,dtype=torch.float32),state=torch.tensor(trip['state'],dtype=torch.long),
                distance_gaps=torch.tensor((np.diff(trip['distances'])-stats['dist_gap'][0])/stats['dist_gap'][1],dtype=torch.float32),
                local_y_raw=torch.tensor(trip['durations'],dtype=torch.float32),
                dist=torch.tensor([(trip['distances'][-1]-stats['dist'][0])/stats['dist'][1]],dtype=torch.float32),
                lengths=torch.tensor(len(coords)),y=torch.tensor([scaler.transform(trip['durations'][-1])],dtype=torch.float32),
                y_raw=torch.tensor([trip['durations'][-1]],dtype=torch.float32))
            row.update({name:torch.tensor(trip[name]) for name in ('uid','weekid','timeid','traj_id','sample_id')})
            rows[index].append(row)
    metadata=DatasetMetadata(data['dataset'],output_dim=1,info=dict(uid_size=len(users),statistics=stats),
        input_layout=dict(kind='planned_route',target_semantics='duration_seconds',coordinates_normalized=True))
    return bundle(rows,metadata,scaler,training,evaluation,seed,
        dict(semantics='complete_trip_time',duration_unit='seconds',train_end=str(cuts[0]),val_end=str(cuts[1])))


def prepare_road(data,training,evaluation,seed,logger=None):
    folder,raw=atomic(data)
    geo=pd.read_csv(folder/(raw.get('geo_file',data['dataset'])+'.geo'))
    rel=pd.read_csv(folder/(raw.get('rel_file',data['dataset'])+'.rel'))
    ids=geo['geo_id'].tolist()
    mapping={identity:index for index,identity in enumerate(ids)}
    positives=np.array([(mapping[a],mapping[b]) for a,b in zip(rel['origin_id'],rel['destination_id'])],dtype=np.int64)
    if len(ids)<2 or len(set(ids))!=len(ids) or len(positives)<3:
        raise ValueError('LINE needs distinct nodes and at least three road edges')
    # Split positive edges before generating negatives so copies cannot cross sets.
    positives=np.unique(positives,axis=0)
    rng=np.random.default_rng(seed)
    positives=positives[rng.permutation(len(positives))]
    cuts=boundaries(len(positives),data['split'])
    forbidden=set(map(tuple,positives))
    degree=np.bincount(positives[:,1],minlength=len(ids)).astype(float)**.75
    rows=[[],[],[]]
    for index,edges in enumerate(np.split(positives,cuts)):
        for origin,destination in edges:
            rows[index].append(dict(I=torch.tensor(origin),J=torch.tensor(destination),y=torch.tensor(1.)))
            candidates=np.array([node for node in range(len(ids)) if (origin,node) not in forbidden])
            if not len(candidates):
                raise ValueError('No valid negative destination exists for an origin')
            weights=degree[candidates]
            probabilities=weights/weights.sum() if weights.sum() else None
            for negative in rng.choice(candidates,size=data['negative_samples'],replace=True,p=probabilities):
                rows[index].append(dict(I=torch.tensor(origin),J=torch.tensor(negative),y=torch.tensor(-1.)))
    metadata=DatasetMetadata(data['dataset'],num_nodes=len(ids),output_dim=1,info=dict(node_ids=ids),
        input_layout=dict(kind='edge_negative_sampling',negative_samples=data['negative_samples']))
    return bundle(rows,metadata,NoneScaler(),training,evaluation,seed,
        dict(semantics='held_out_positive_edges',positive_edges=[len(part) for part in np.split(positives,cuts)]))
