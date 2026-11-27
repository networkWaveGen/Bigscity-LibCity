"""Small atomic map-matching adapter; no training loaders or generated pickle cache."""
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

from libcity.data.bundle import DataBundle,DatasetMetadata
from libcity.utils.GPS_utils import dist,angle2radian


def prepare_map_matching(data_config,training_config,evaluation_config,seed,logger=None):
    if data_config['cache']:
        raise ValueError('Map-matching adapter requires data.cache=false')
    folder=Path(data_config['data_dir'])/data_config['dataset']
    raw=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    raw={**raw,**raw.get('info',{})}
    name=data_config['dataset']
    geo=pd.read_csv(folder/(raw.get('geo_file',name)+'.geo'))
    rel=pd.read_csv(folder/(raw.get('rel_file',name)+'.rel'))
    users=set(pd.read_csv(folder/(raw.get('usr_file',name)+'.usr'))['usr_id'].tolist())
    observations=pd.read_csv(folder/(raw.get('dyna_file',name)+'.dyna'))
    truth=pd.read_csv(folder/(raw.get('truth_file',name+'_truth')+'.dyna'))
    ids=geo['geo_id'].tolist()
    if len(set(ids))!=len(ids):
        raise ValueError('Map geometry contains duplicate road IDs')
    indices={identity:index for index,identity in enumerate(ids)}
    parents=list(range(2*len(ids)))
    def find(index):
        while parents[index]!=index:
            parents[index]=parents[parents[index]]
            index=parents[index]
        return index
    for road in rel.to_dict('records'):
        parents[find(indices[road['origin_id']])]=find(indices[road['destination_id']]+len(ids))
    graph=nx.DiGraph()
    for road in geo.to_dict('records'):
        identity=road['geo_id']
        coordinates=json.loads(road['coordinates'])
        if np.asarray(coordinates).shape!=(2,2) or not np.isfinite(coordinates).all():
            raise ValueError('Map-matching atomic adapter requires finite two-endpoint LineString segments')
        start,end=coordinates
        origin,destination=find(indices[identity]+len(ids)),find(indices[identity])
        for vertex,coordinate in ((origin,start),(destination,end)):
            if vertex not in graph:
                graph.add_node(vertex,lon=float(coordinate[0]),lat=float(coordinate[1]))
        if graph.has_edge(origin,destination):
            raise ValueError('Map-matching adapter cannot collapse parallel road segments into a DiGraph')
        distance=road.get('distance',dist(angle2radian(start[1]),angle2radian(start[0]),
                                        angle2radian(end[1]),angle2radian(end[0])))
        if not np.isfinite(distance) or distance<=0:
            raise ValueError('Road segment distance must be finite and positive')
        attributes=dict(geo_id=int(identity),distance=float(distance))
        if 'speed' in road:
            attributes['speed']=float(road['speed'])
        graph.add_edge(origin,destination,**attributes)
    with_time=raw.get('with_time',False)
    delta_time=raw.get('delta_time',True)
    if type(with_time) is not bool or type(delta_time) is not bool:
        raise TypeError('with_time and delta_time metadata must be booleans')
    trajectories={}
    for row in observations.to_dict('records'):
        user,trajectory=int(row['entity_id']),int(row.get('traj_id',0))
        if user not in users:
            raise ValueError('Trajectory user does not appear in .usr file')
        coordinate=json.loads(row['coordinates'])
        if np.asarray(coordinate).shape!=(2,) or not np.isfinite(coordinate).all():
            raise ValueError('GPS coordinates must be finite longitude/latitude pairs')
        values=[int(row['dyna_id']),*coordinate]
        if with_time:
            values.append(pd.Timestamp(row['time']).to_pydatetime())
        trajectories.setdefault(user,{}).setdefault(trajectory,[]).append(values)
    for groups in trajectories.values():
        for trajectory,values in groups.items():
            if with_time and delta_time:
                first=values[0][3]
                for value in values:
                    value[3]=(value[3]-first).total_seconds()
            groups[trajectory]=np.asarray(values)
    routes={}
    for row in truth.to_dict('records'):
        user,trajectory=int(row['entity_id']),int(row.get('traj_id',0))
        location=int(row['location'])
        if user not in trajectories or trajectory not in trajectories[user] or location not in indices:
            raise ValueError('Truth route does not match users/trajectories/roads')
        routes.setdefault(user,{}).setdefault(trajectory,[]).append([int(row['dyna_id']),location])
    for user,groups in trajectories.items():
        for trajectory in groups:
            if not routes.get(user,{}).get(trajectory):
                raise ValueError('Every evaluated trajectory requires a nonempty truth route')
            routes[user][trajectory]=np.asarray(routes[user][trajectory],dtype=np.int64)
    if not trajectories or not len(graph):
        raise ValueError('Map-matching data and road network must not be empty')
    features=dict(with_time=with_time,delta_time=delta_time,with_rd_speed='speed' in geo.columns)
    metadata=DatasetMetadata(name,num_nodes=len(graph),info=dict(raw=raw,features=features))
    test=dict(rd_nwk=nx.freeze(graph),trajectory=trajectories,route=routes)
    summary=dict(train=0,val=0,test=sum(len(groups) for groups in trajectories.values()),
                 samples=len(observations),semantics='direct_evaluation')
    if logger:
        logger.info('Prepared map-matching atomic data: %s',summary)
    return DataBundle(None,None,test,metadata,split_summary=summary)
