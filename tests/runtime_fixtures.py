"""Small atomic files shared by runtime acceptance and numerical comparisons."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_fixture(root, name='Tiny', steps=48, nodes=3):
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'config.json').write_text(json.dumps({
        'geo_file': name, 'rel_file': name, 'data_files': name,
        'data_col': ['speed'], 'output_dim': 1, 'time_intervals': 300}), encoding='utf-8')
    pd.DataFrame(dict(geo_id=range(nodes), type=['Point'] * nodes,
                      coordinates=[f'[{i},0]' for i in range(nodes)])).to_csv(folder / f'{name}.geo', index=False)
    pd.DataFrame(dict(rel_id=range(nodes), type=['geo'] * nodes, origin_id=range(nodes),
                      destination_id=[(i + 1) % nodes for i in range(nodes)],
                      distance=[1.] * nodes)).to_csv(folder / f'{name}.rel', index=False)
    times = pd.date_range('2024-01-01', periods=steps, freq='5min').strftime('%Y-%m-%dT%H:%M:%SZ')
    rows = [dict(dyna_id=node * steps + t, type='state', time=times[t], entity_id=node,
                 speed=0. if (t, node) == (6, 1) else 10 + node + float(np.sin(t / 4)))
            for node in range(nodes) for t in range(steps)]
    pd.DataFrame(rows).to_csv(folder / f'{name}.dyna', index=False)
    return folder
