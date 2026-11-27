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
