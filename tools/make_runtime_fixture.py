"""Copy a published config and create its atomic sample data outside the source tree."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from libcity.testing.fixtures import atomic_fixture,crann_fixture,map_fixture,task_fixture,road_feature_fixture,multiscale_fixture,tgclstm_fixture


EXAMPLES={'traffic_state_pred':'gru','traj_loc_pred':'rnn','eta':'deeptte',
          'road_representation':'line','map_matching':'hmmm'}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--task',choices=EXAMPLES)
    parser.add_argument('--model',choices=('GRU','CRANN','ChebConv','GAT','ASTGCN','MSTGCN','MegaCRN','TESTAM','ToGCN','SimST','STNN','TGCLSTM','STTN'),help='Optional named model example')
    args=parser.parse_args(argv)
    selected_task = 'road_representation' if args.model in ('ChebConv','GAT') else 'traffic_state_pred'
    if args.model and args.task and args.task!=selected_task:
        parser.error('--model and --task select different tasks')
    args.task = args.task or selected_task
    directory=args.directory.expanduser().resolve()
    directory.mkdir(parents=True,exist_ok=True)
    fixture=(road_feature_fixture if args.model in ('ChebConv','GAT') else
        (tgclstm_fixture if args.model=='TGCLSTM' else crann_fixture if args.model=='CRANN' else multiscale_fixture if args.model in ('ASTGCN','MSTGCN') else atomic_fixture) if args.task=='traffic_state_pred' else
        map_fixture if args.task=='map_matching' else task_fixture)
    fixture(directory/'data')
    name=args.model.lower() if args.model else EXAMPLES[args.task]
    config=json.loads((ROOT/'examples/configs'/f'{name}.json').read_text(encoding='utf-8'))
    path=directory/f'{name}.json'
    path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    print(path)
    return path


if __name__=='__main__':
    main()
