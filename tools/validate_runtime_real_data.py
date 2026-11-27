"""Convert a METR-LA raw NumPy archive and record a reproducible CPU short run.

Download URL is the one used by PyTorch Geometric Temporal's METRLADatasetLoader.
The archive is data only: only two known .npy members are loaded, allow_pickle=False.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import zipfile
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from libcity.pipeline.experiment import run_experiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--steps',type=int,default=2016)
    args=parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        values=np.load(archive.open('node_values.npy'),allow_pickle=False)
        adjacency=np.load(archive.open('adj_mat.npy'),allow_pickle=False)
    full_shape=values.shape
    if values.ndim!=3 or values.shape[1]!=207 or adjacency.shape!=(207,207) or not 72<=args.steps<=len(values):
        raise ValueError('Expected original METR-LA arrays and a valid time subset')
    values=values[:args.steps,:,0]
    destination=args.directory.resolve()
    folder=destination/'data/METR_LA'
    folder.mkdir(parents=True,exist_ok=True)
    nodes=values.shape[1]
    # Clock comes from the original dataset's 2012-03-01 start and 5-minute grid.
    times=pd.date_range('2012-03-01',periods=len(values),freq='5min').strftime('%Y-%m-%dT%H:%M:%SZ')
    pd.DataFrame(dict(geo_id=range(nodes),type=['Point']*nodes,coordinates=['[]']*nodes)).to_csv(folder/'METR_LA.geo',index=False)
    origins,destinations=np.nonzero(adjacency)
    pd.DataFrame(dict(rel_id=range(len(origins)),type='geo',origin_id=origins,destination_id=destinations,
                      weight=adjacency[origins,destinations])).to_csv(folder/'METR_LA.rel',index=False)
    pd.DataFrame(dict(dyna_id=np.arange(values.size),type='state',time=np.tile(times,nodes),
        entity_id=np.repeat(np.arange(nodes),len(values)),speed=values.T.reshape(-1))).to_csv(folder/'METR_LA.dyna',index=False)
    metadata=dict(geo_file='METR_LA',rel_file='METR_LA',data_files='METR_LA',data_col=['speed'],
                  output_dim=1,time_intervals=300,weight_col='weight')
    (folder/'config.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    config=dict(data=dict(dataset='METR_LA',data_dir=str(destination/'data'),cache_dir=str(destination/'cache'),
        cache=False,pad_training=False,input_window=12,output_window=12,split=dict(train=.7,val=.1)),
        model=dict(name='GRU',params=dict(hidden_size=16)),training=dict(epochs=1,batch_size=32),
        runtime=dict(device='cpu',seed=17,exp_id='metr-la-gru',output_dir=str(destination/'runs')))
    path=destination/'gru.json'
    path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    result=run_experiment('train',path)
    evaluation=run_experiment('evaluate',path,checkpoint=result.artifacts['best'],overrides={'runtime':{'exp_id':'metr-la-evaluate'}})
    for key,value in evaluation.metrics.items():
        if not np.isclose(value,result.metrics[key]):
            raise AssertionError(f'Independent evaluation disagrees for {key}')
    info=json.loads(result.artifacts['runtime_info'].read_text(encoding='utf-8'))
    record=dict(passed=True,source_loader='https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/dataset/metr_la.py',
        source_url='https://anl.app.box.com/shared/static/plgsv3te0akmqluiuqva34su60nn93c2',
        full_shape=list(full_shape),used_steps=len(values),used_nodes=nodes,subset='first seven days' if len(values)==2016 else 'prefix time subset',
        coordinate_note='Sensor coordinates are not in this archive; empty geometry placeholders are unused by GRU.',
        time_origin='2012-03-01T00:00:00, regular five-minute sampling',config=config,
        runtime=info,metrics=result.metrics,independent_evaluation=evaluation.metrics,
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,report=str(args.report),metrics=result.metrics)))


if __name__=='__main__':
    main()
