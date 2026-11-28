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


def dgcn_fixture(root,name='Tiny'):
    return atomic_fixture(root,name=name,steps=800,interval_minutes=60)


def periodic_fixture(root,name='PeriodicTiny',steps=400,nodes=4,features=2,outputs=2,interval_minutes=60):
    folder = atomic_fixture(root,name=name,steps=steps,nodes=nodes,interval_minutes=interval_minutes)
    frame = pd.read_csv(folder/f'{name}.dyna').rename(columns={'speed':'value_0'})
    for index in range(1,features):
        frame[f'value_{index}'] = 20+index*7+frame['entity_id']*.5+2*np.cos((frame['dyna_id']%steps)/(index+3))
    frame.to_csv(folder/f'{name}.dyna',index=False)
    records = [(origin,destination,1+3*((destination-origin)%nodes-1))
        for origin in range(nodes) for destination in range(nodes) if origin!=destination]
    pd.DataFrame(dict(rel_id=range(len(records)),type='geo',origin_id=[row[0] for row in records],
        destination_id=[row[1] for row in records],distance=[row[2] for row in records])).to_csv(folder/f'{name}.rel',index=False)
    path = folder/'config.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata.update(data_col=[f'value_{index}' for index in range(features)],output_dim=outputs,weight_col='distance')
    path.write_text(json.dumps(metadata),encoding='utf-8')
    return folder


def od_fixture(root,name='ODTiny',steps=48,nodes=3,features=2,outputs=1,grid_shape=None,extension=None):
    folder = Path(root)/name
    folder.mkdir(parents=True,exist_ok=True)
    nodes = grid_shape[0]*grid_shape[1] if grid_shape else nodes
    order = np.random.default_rng(19).permutation(nodes)
    ids = 53+order*7
    geo = pd.DataFrame(dict(geo_id=ids,type='Polygon' if grid_shape else 'Point',coordinates='[]'))
    if grid_shape:
        geo['row_id'],geo['column_id'] = order//grid_shape[1],order%grid_shape[1]
    geo.to_csv(folder/f'{name}.geo',index=False)
    rel = [dict(rel_id=i*nodes+j,type='geo',origin_id=int(ids[i]),destination_id=int(ids[j]),distance=1+2*i+j)
        for i in range(nodes) for j in range(nodes) if i!=j]
    pd.DataFrame(rel,columns=['rel_id','type','origin_id','destination_id','distance']).to_csv(folder/f'{name}.rel',index=False)
    clock = pd.date_range('2024-01-01',periods=steps,freq='5min',tz='UTC')
    records = []
    for time in range(steps):
        for origin in range(nodes):
            for destination in range(nodes):
                row = dict(dyna_id=len(records),type='state',time=clock[time].strftime('%Y-%m-%dT%H:%M:%SZ'))
                if grid_shape:
                    row.update(origin_row_id=int(geo.iloc[origin]['row_id']),origin_column_id=int(geo.iloc[origin]['column_id']),
                        destination_row_id=int(geo.iloc[destination]['row_id']),destination_column_id=int(geo.iloc[destination]['column_id']))
                else:
                    row.update(origin_id=int(ids[origin]),destination_id=int(ids[destination]))
                row.update({f'flow_{feature}':0. if (time,origin,destination,feature)==(6,1,2,0) else
                    10+time*.1+origin*3+destination*.7+feature*100 for feature in range(features)})
                records.append(row)
    suffix = extension or ('.gridod' if grid_shape else '.od')
    pd.DataFrame(records).sample(frac=1,random_state=17).to_csv(folder/f'{name}{suffix}',index=False)
    pd.DataFrame(dict(ext_id=range(steps),time=clock.strftime('%Y-%m-%dT%H:%M:%SZ'),weather=np.arange(steps)*2+.5,
        holiday=np.arange(steps)%7)).sample(frac=1,random_state=21).to_csv(folder/f'{name}.ext',index=False)
    (folder/'config.json').write_text(json.dumps(dict(geo_file=name,rel_file=name,data_files=name,
        data_col=[f'flow_{feature}' for feature in range(features)],output_dim=outputs,time_intervals=300,
        ext_file=name,ext_col=['weather','holiday'],weight_col='distance')),encoding='utf-8')
    return folder


def geml_fixture(root):
    return od_fixture(root,features=1,outputs=1)


def cstn_fixture(root):
    return od_fixture(root,features=1,outputs=1,grid_shape=(2,2))


def dsan_fixture(root):
    return grid_fixture(root,steps=60)


def d2stgnn_fixture(root,name='DecoupleTiny',steps=60):
    return stmgat_fixture(root,name,steps)


def stmgat_fixture(root,name='GATTiny',steps=60):
    folder=periodic_fixture(root,name,steps,nodes=4,features=3,outputs=2)
    ids=np.array([71,3,108,400]);path=folder/f'{name}.geo';geo=pd.read_csv(path);geo['geo_id']=ids
    geo.sample(frac=1,random_state=23).to_csv(path,index=False)
    path=folder/f'{name}.dyna';frame=pd.read_csv(path);frame['entity_id']=frame['entity_id'].map(dict(enumerate(ids)))
    frame.to_csv(path,index=False)
    pd.DataFrame(dict(rel_id=range(3),type='geo',origin_id=ids[:3],destination_id=ids[[1,2,0]],
        distance=[1.,2.,3.])).to_csv(folder/f'{name}.rel',index=False)
    return folder


def hiest_fixture(root,name='HierarchyTiny',steps=60):
    folder=atomic_fixture(root,name,steps,nodes=7)
    ids=np.array([71,3,108,19,250,6,400])
    path=folder/f'{name}.geo';geo=pd.read_csv(path);geo['geo_id']=ids
    geo.sample(frac=1,random_state=23).to_csv(path,index=False)
    path=folder/f'{name}.dyna';frame=pd.read_csv(path).rename(columns={'speed':'value_0'})
    frame['value_1']=20+frame['entity_id']*.3+np.cos((frame['dyna_id']%steps)/3)
    frame['entity_id']=frame['entity_id'].map(dict(enumerate(ids)))
    frame.to_csv(path,index=False)
    edges=[(0,1),(1,2),(2,0),(2,3),(3,4),(4,2),(4,5)]
    pairs=[pair for edge in edges for pair in (edge,edge[::-1])]
    pd.DataFrame(dict(rel_id=range(len(pairs)),type='geo',origin_id=[int(ids[a]) for a,b in pairs],
        destination_id=[int(ids[b]) for a,b in pairs],distance=1.)).to_csv(folder/f'{name}.rel',index=False)
    path=folder/'config.json';raw=json.loads(path.read_text(encoding='utf-8'))
    raw.update(data_col=['value_0','value_1'],output_dim=2)
    path.write_text(json.dumps(raw),encoding='utf-8')
    return folder


def stdn_fixture(root,name='STDNTiny',steps=160,rows=2,columns=3,features=2,outputs=2):
    folder=grid_fixture(root,name,steps,rows,columns,features,outputs)
    clock=pd.date_range('2024-01-01',periods=steps,freq='h',tz='UTC')
    records=[]
    for step in range(steps):
        for origin in range(rows*columns):
            for destination in range(rows*columns):
                records.append(dict(dyna_id=len(records),type='state',time=clock[step].strftime('%Y-%m-%dT%H:%M:%SZ'),
                    origin_row_id=origin//columns,origin_column_id=origin%columns,
                    destination_row_id=destination//columns,destination_column_id=destination%columns,
                    same_slot=0. if (step,origin,destination)==(50,1,2) else 1+step*.1+origin*3+destination*.7,
                    next_slot=2+step*.2+origin*.3+destination*2))
    pd.DataFrame(records).sample(frac=1,random_state=23).to_csv(folder/f'{name}.gridod',index=False)
    path=folder/'config.json';raw=json.loads(path.read_text(encoding='utf-8'))
    raw.update(flow_files=name,flow_col=['same_slot','next_slot'])
    path.write_text(json.dumps(raw),encoding='utf-8')
    return folder


def risk_fixture(root,name='RiskTiny',steps=96,poi=True):
    folder = grid_fixture(root,name=name,steps=steps,rows=2,columns=3,features=3,outputs=1)
    ids = [71,3,108]
    cells = [(1,2),(0,0),(0,2)]
    pd.DataFrame(dict(geo_id=ids,row_id=[value[0] for value in cells],column_id=[value[1] for value in cells],
        risk_mask=[.5,1.,2.],type='Polygon',coordinates='[]')).to_csv(folder/f'{name}.geo',index=False)
    records = [dict(rel_id=i*3+j,type='geo',origin_id=ids[i],destination_id=ids[j],
        road=float(i==j or abs(i-j)==1),risk=.3*(i+1)/(j+1),poi=1. if i==j else .2+.1*i)
        for i in range(3) for j in range(3)]
    relation = pd.DataFrame(records)
    if not poi: relation = relation.drop(columns='poi')
    relation.sample(frac=1,random_state=13).to_csv(folder/f'{name}.rel',index=False)
    frame = pd.read_csv(folder/f'{name}.grid')
    frame['value_2'] = pd.to_datetime(frame['time'],utc=True).dt.dayofweek
    frame.to_csv(folder/f'{name}.grid',index=False)
    path = folder/'config.json';raw=json.loads(path.read_text(encoding='utf-8'))
    raw.update(grid_len_row=2,grid_len_column=3,rel_file=name,weight_col=['road','risk']+(['poi'] if poi else []),
        graph_input_col=['value_0','value_1'],target_time_col=['value_2'],
        risk_thresholds=[0.,20.,28.],risk_weights=[.05,.2,.25,.5])
    path.write_text(json.dumps(raw),encoding='utf-8')
    return folder


def dmstgcn_fixture(root,name='Tiny'):
    return periodic_fixture(root,name=name,steps=60,nodes=4,features=2,outputs=1)


def reslstm_fixture(root,name='PeriodicTiny'):
    return periodic_fixture(root,name=name,steps=800)


def grid_fixture(root, name='GridTiny', steps=400, rows=2, columns=3, features=3, outputs=2):
    folder = Path(root)/name
    folder.mkdir(parents=True,exist_ok=True)
    order = np.random.default_rng(19).permutation(rows*columns)
    pd.DataFrame(dict(geo_id=100+order,type='Polygon',coordinates='[]',row_id=order//columns,
        column_id=order%columns)).to_csv(folder/f'{name}.geo',index=False)
    clock = pd.date_range('2024-01-01',periods=steps,freq='h',tz='UTC')
    frame = pd.DataFrame([dict(dyna_id=step*rows*columns+node,type='state',
        time=clock[step].strftime('%Y-%m-%dT%H:%M:%SZ'),row_id=node//columns,column_id=node%columns,
        **{f'value_{feature}':20+node*2+feature*4+np.sin(step/12) for feature in range(features)})
        for step in range(steps) for node in range(rows*columns)])
    frame.loc[(frame['dyna_id']%37)==0,'value_0'] = 0.
    frame.sample(frac=1,random_state=11).to_csv(folder/f'{name}.grid',index=False)
    pd.DataFrame(dict(ext_id=range(steps),time=clock.strftime('%Y-%m-%dT%H:%M:%SZ'),
        temperature=10+np.arange(steps)*.01,holiday=(clock.day_of_week>=5).astype(float))).to_csv(folder/f'{name}.ext',index=False)
    (folder/'config.json').write_text(json.dumps(dict(geo_file=name,data_files=name,ext_file=name,
        time_intervals=3600,data_col=[f'value_{index}' for index in range(features)],output_dim=outputs)),encoding='utf-8')
    return folder


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


def ttpnet_fixture(root,name='TaskTiny'):
    folder = task_fixture(root,name)
    geo = pd.read_csv(folder/f'{name}.geo')
    geo['embeddings'] = [json.dumps((np.sin(np.arange(32)+index)*.1).tolist()) for index in range(len(geo))]
    geo.to_csv(folder/f'{name}.geo',index=False)
    frame = pd.read_csv(folder/f'{name}.dyna')
    for name_column,offset,width in (('speeds',0,4),('speeds_relevant1',3,4),('speeds_relevant2',7,4),('speeds_long',1,7)):
        frame[name_column] = [json.dumps([float(20+offset+(index+slot)%15) if (index+slot)%19 else 0.
            for slot in range(width)]) for index in range(len(frame))]
    frame['grid_len'] = .1+np.arange(len(frame))%5*.01
    frame['traffic_snapshot_time'] = (pd.to_datetime(frame.groupby(['entity_id','traj_id'])['time'].transform('min'),utc=True)
        -pd.Timedelta(minutes=15)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    frame.to_csv(folder/f'{name}.dyna',index=False)
    return folder


def serm_fixture(root,name='TaskTiny'):
    folder = task_fixture(root,name)
    geo = pd.read_csv(folder/f'{name}.geo')
    geo['venue_category_name'] = ['Coffee Shop','Park','Coffee coffee unknown','',
        'unknown unseen','Shop','park food','food']
    geo.to_csv(folder/f'{name}.geo',index=False)
    vectors = Path(root)/'word_vec'
    vectors.mkdir(parents=True,exist_ok=True)
    (vectors/'glove.fixture.txt').write_text('unused 0 0 0\ncoffee 1 2 3\nshop -1 0 2\npark .5 .25 .75\nfood 0 1 -1\n',encoding='utf-8')
    path = folder/'config.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    metadata.update(word_vector_file='../word_vec/glove.fixture.txt',semantic_column='venue_category_name')
    path.write_text(json.dumps(metadata),encoding='utf-8')
    return folder
