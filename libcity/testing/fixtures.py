"""Small atomic files shared by runtime acceptance and numerical comparisons."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_fixture(root, name='Tiny', steps=48, nodes=3,interval_minutes=5):
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'config.json').write_text(json.dumps({
        'geo_file': name, 'rel_file': name, 'data_files': name,
        'data_col': ['speed'], 'output_dim': 1, 'time_intervals': interval_minutes*60}), encoding='utf-8')
    pd.DataFrame(dict(geo_id=range(nodes), type=['Point'] * nodes,
                      coordinates=[f'[{i},0]' for i in range(nodes)])).to_csv(folder / f'{name}.geo', index=False)
    pd.DataFrame(dict(rel_id=range(nodes), type=['geo'] * nodes, origin_id=range(nodes),
                      destination_id=[(i + 1) % nodes for i in range(nodes)],
                      distance=[1.] * nodes)).to_csv(folder / f'{name}.rel', index=False)
    times = pd.date_range('2024-01-01', periods=steps, freq=f'{interval_minutes}min').strftime('%Y-%m-%dT%H:%M:%SZ')
    rows = [dict(dyna_id=node * steps + t, type='state', time=times[t], entity_id=node,
                 speed=0. if (t, node) == (6, 1) else 10 + node + float(np.sin(t / 4)))
            for node in range(nodes) for t in range(steps)]
    pd.DataFrame(rows).to_csv(folder / f'{name}.dyna', index=False)
    return folder


def tgclstm_fixture(root,name='Tiny',steps=60,nodes=3):
    folder = atomic_fixture(root,name=name,steps=steps,nodes=nodes)
    edges = pd.read_csv(folder/f'{name}.rel')
    for minutes in (5,10,15,20,25):
        edges[f'FFR_{minutes}min'] = (np.arange(nodes)+1)/nodes*minutes/25
    edges.to_csv(folder/f'{name}.rel',index=False)
    path = folder/'config.json'
    metadata = json.loads(path.read_text())
    metadata['weight_col'] = 'distance'
    path.write_text(json.dumps(metadata),encoding='utf-8')
    return folder


def multiscale_fixture(root, name='Tiny'):
    return atomic_fixture(root,name=name,steps=400,interval_minutes=60)


def road_feature_fixture(root, name='RoadFeatures', nodes=12):
    folder = Path(root)/name
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'config.json').write_text(json.dumps(dict(geo_file=name,rel_file=name)),encoding='utf-8')
    ids = np.arange(100,100+nodes)
    pd.DataFrame(dict(geo_id=ids,type='LineString',coordinates='[[116,39],[116.001,39]]',
        length=np.arange(nodes)*5.+20,maxspeed=np.full(nodes,40.),width=np.arange(nodes)%3+3.,
        lanes=np.arange(nodes)%2+1,highway=['primary' if index%2 else 'secondary' for index in range(nodes)]
    )).to_csv(folder/f'{name}.geo',index=False)
    pairs = [(int(ids[index]),int(ids[(index+step)%nodes])) for index in range(nodes) for step in (1,2)]
    pd.DataFrame(dict(rel_id=range(len(pairs)),type='geo',origin_id=[a for a,b in pairs],
        destination_id=[b for a,b in pairs])).to_csv(folder/f'{name}.rel',index=False)
    return folder


def crann_fixture(root, name='Tiny', steps=100):
    folder = atomic_fixture(root, name=name, steps=steps)
    times = pd.read_csv(folder/f'{name}.dyna')['time'].drop_duplicates().to_numpy()
    pd.DataFrame(dict(ext_id=range(steps), time=times, temperature=np.arange(steps)*.01,
                      weather=np.arange(steps)*.1, holiday=np.arange(steps)%2)).to_csv(folder/f'{name}.ext', index=False)
    return folder


def map_fixture(root,name='RoadTiny',with_time=False):
    folder=Path(root)/name
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'config.json').write_text(json.dumps(dict(geo_file=name,rel_file=name,dyna_file=name,usr_file=name,
        truth_file=name+'_truth',with_time=with_time,delta_time=True)),encoding='utf-8')
    pd.DataFrame(dict(geo_id=[0,1,2],type=['LineString']*3,
        coordinates=[json.dumps([[116+i*.001,39.],[116+(i+1)*.001,39.]]) for i in range(3)])).to_csv(folder/f'{name}.geo',index=False)
    pd.DataFrame(dict(rel_id=[0,1],type=['geo']*2,origin_id=[0,1],destination_id=[1,2])).to_csv(folder/f'{name}.rel',index=False)
    pd.DataFrame({'usr_id':[7]}).to_csv(folder/f'{name}.usr',index=False)
    rows=[]
    truth=[]
    for group in range(2):
        for index,position in enumerate((.0002,.0007,.0013,.0017,.0023,.0028)):
            rows.append(dict(dyna_id=group*6+index,type='trajectory',entity_id=7,traj_id=101+group,
                time=(pd.Timestamp('2024-01-01')+pd.Timedelta(seconds=30*(group*6+index))).isoformat()+'Z',
                coordinates=json.dumps([116+position,39.00001])))
        for road in range(3):
            truth.append(dict(dyna_id=group*3+road,type='trajectory',entity_id=7,traj_id=101+group,location=road))
    pd.DataFrame(rows).to_csv(folder/f'{name}.dyna',index=False)
    pd.DataFrame(truth).to_csv(folder/f'{name}_truth.dyna',index=False)
    return folder


def task_fixture(root,name='TaskTiny'):
    folder=Path(root)/name
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'config.json').write_text(json.dumps(dict(geo_file=name,rel_file=name,dyna_file=name,usr_file=name)),encoding='utf-8')
    pd.DataFrame(dict(geo_id=range(10,18),type=['Point']*8,coordinates=['[116,39]']*8)).to_csv(folder/f'{name}.geo',index=False)
    pairs=[(10+origin,10+(origin+step)%8) for origin in range(8) for step in (1,2)]
    pd.DataFrame(dict(rel_id=range(len(pairs)),type='geo',origin_id=[a for a,b in pairs],destination_id=[b for a,b in pairs])).to_csv(folder/f'{name}.rel',index=False)
    pd.DataFrame({'usr_id':[100,200]}).to_csv(folder/f'{name}.usr',index=False)
    rows=[]
    for trip in range(20):
        start=pd.Timestamp('2024-01-01',tz='UTC')+pd.Timedelta(days=trip)
        for point in range(3+trip%4):
            rows.append(dict(dyna_id=len(rows),type='trajectory',entity_id=100+100*(trip%2),traj_id=trip,
                time=(start+pd.Timedelta(seconds=point*(60+trip))).strftime('%Y-%m-%dT%H:%M:%SZ'),
                location=10+(trip+point)%8,coordinates=json.dumps([116+point*.001,39+trip*.0001]),
                current_dis=point*.1,current_state=point%2))
    pd.DataFrame(rows).to_csv(folder/f'{name}.dyna',index=False)
    return folder


def strnn_fixture(root,name='TaskTiny'):
    folder = task_fixture(root,name)
    geo = pd.read_csv(folder/f'{name}.geo')
    geo['coordinates'] = [json.dumps([116+index*.001,39+(index%3)*.0005]) for index in range(len(geo))]
    geo.to_csv(folder/f'{name}.geo',index=False)
    return folder
