"""Copy a published config and create its atomic sample data outside the source tree."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from libcity.testing.fixtures import atomic_fixture,crann_fixture,map_fixture,task_fixture,road_feature_fixture,multiscale_fixture,tgclstm_fixture,strnn_fixture,serm_fixture,ttpnet_fixture,grid_fixture,periodic_fixture,reslstm_fixture,dgcn_fixture,dmstgcn_fixture,geml_fixture,cstn_fixture,dsan_fixture,risk_fixture,stdn_fixture
from libcity.model.runtime_registry import model_spec


EXAMPLES={'traffic_state_pred':'gru','traj_loc_pred':'rnn','eta':'deeptte',
          'road_representation':'line','map_matching':'hmmm'}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--task',choices=EXAMPLES)
    parser.add_argument('--model',choices=('GRU','STDN','GSNet','DSAN','CSTN','GEML','DMSTGCN','CCRNN','RGSL','GTS','HGCN','DGCN','STPGCN','STSGCN','DKFN','ResLSTM','CRANN','ChebConv','GAT','GeomGCN','STResNet','ACFM','CONVGCN','MultiSTGCnet','ASTGCN','MSTGCN','MegaCRN','TESTAM','ToGCN','SimST','STNN','TGCLSTM','STTN','TimeMixer','STG2Seq','DeepMove','STRNN','HSTLSTM','ATSTLSTM','SERM','STAN','CARA','LSTPM','GeoSAN','TTPNet','Node2Vec','DeepWalk'),help='Optional named model example')
    args=parser.parse_args(argv)
    selected_task = model_spec(args.model).task if args.model else args.task or 'traffic_state_pred'
    if args.model and args.task and args.task!=selected_task:
        parser.error('--model and --task select different tasks')
    args.task = args.task or selected_task
    directory=args.directory.expanduser().resolve()
    directory.mkdir(parents=True,exist_ok=True)
    fixture=(stdn_fixture if args.model=='STDN' else risk_fixture if args.model=='GSNet' else dsan_fixture if args.model=='DSAN' else cstn_fixture if args.model=='CSTN' else geml_fixture if args.model=='GEML' else dmstgcn_fixture if args.model=='DMSTGCN' else dgcn_fixture if args.model=='DGCN' else reslstm_fixture if args.model=='ResLSTM' else periodic_fixture if args.model in ('CONVGCN','MultiSTGCnet') else grid_fixture if args.model in ('STResNet','ACFM') else road_feature_fixture if args.model in ('ChebConv','GAT','GeomGCN') else
        (tgclstm_fixture if args.model=='TGCLSTM' else crann_fixture if args.model=='CRANN' else multiscale_fixture if args.model in ('ASTGCN','MSTGCN') else atomic_fixture) if args.task=='traffic_state_pred' else
        map_fixture if args.task=='map_matching' else ttpnet_fixture if args.model=='TTPNet' else serm_fixture if args.model=='SERM' else strnn_fixture if args.model in ('STRNN','HSTLSTM','ATSTLSTM','STAN','CARA','LSTPM','GeoSAN') else task_fixture)
    fixture(directory/'data')
    name=args.model.lower() if args.model else EXAMPLES[args.task]
    config=json.loads((ROOT/'examples/configs'/f'{name}.json').read_text(encoding='utf-8'))
    path=directory/f'{name}.json'
    path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    print(path)
    return path


if __name__=='__main__':
    main()
