"""Source CLI for finite search; every trial uses the common experiment runtime."""
import argparse
import json
from libcity.pipeline.runtime_tuning import run_search


def main(argv=None):
    parser=argparse.ArgumentParser(description='Tune LibCity on validation data, evaluate the winner once')
    parser.add_argument('--config',required=True)
    parser.add_argument('--space',required=True,help='JSON object of dotted config keys and finite choices')
    parser.add_argument('--algorithm',choices=('grid','random'),default='grid')
    parser.add_argument('--max-trials',type=int,default=10)
    args=parser.parse_args(argv)
    result=run_search(args.config,args.space,algorithm=args.algorithm,max_trials=args.max_trials)
    print(json.dumps(dict(metrics=result.metrics,artifacts={key:str(path) for key,path in result.artifacts.items()})))
    return result


if __name__=='__main__':
    main()
