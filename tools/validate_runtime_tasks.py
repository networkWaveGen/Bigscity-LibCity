"""Run all five task examples via actual CLI processes from outside the repository."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--minimal-env',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    directory=args.directory.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    optional={name:importlib.util.find_spec(name) is not None for name in ('tensorboard','einops','ray','hyperopt','dgl','torchtext')}
    if args.minimal_env and any(optional.values()):
        raise AssertionError(optional)
    records=[]
    def run(arguments,*,parse=True):
        result=subprocess.run([sys.executable,'-B',*map(str,arguments)],cwd=directory,
            capture_output=True,text=True,encoding='utf-8',errors='replace')
        records.append(dict(arguments=list(map(str,arguments)),exit_code=result.returncode))
        if result.returncode:
            raise RuntimeError(result.stdout+'\n'+result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1]) if parse else None
    tasks={}
    for task,filename in dict(traffic_state_pred='gru',traj_loc_pred='rnn',eta='deeptte',road_representation='line',map_matching='hmmm').items():
        target=directory/task
        run([root/'tools/make_runtime_fixture.py','--directory',target,'--task',task],parse=False)
        config=target/(filename+'.json')
        if task=='map_matching':
            result=run([root/'run_model.py','evaluate','--config',config,'--set','runtime.exp_id=direct'])
            assert result['metrics']==dict(RMF=0.,AN=1.,AL=1.)
            tasks[task]=dict(metrics=result['metrics'],checkpoint_required=False)
            continue
        trained=run([root/'run_model.py','train','--config',config,'--set','runtime.exp_id=train'])
        evaluated=run([root/'run_model.py','evaluate','--config',config,'--checkpoint',trained['artifacts']['best'],'--set','runtime.exp_id=eval'])
        for key,value in evaluated['metrics'].items():
            if abs(value-trained['metrics'][key])>1e-7:
                raise AssertionError(f'{task}/{key}: independent evaluation differs')
        resumed=run([root/'run_model.py','train','--config',config,'--resume',trained['artifacts']['last'],
                     '--set','training.epochs=3','--set','runtime.exp_id=resumed'])
        state=torch.load(resumed['artifacts']['last'],weights_only=True)
        assert state['train_state']['epoch']==3
        assert not (target/'runs/eval/metrics.jsonl').exists()
        tasks[task]=dict(metrics=evaluated['metrics'],resumed_epoch=3,evaluate_has_no_training_writer=True)
    space=directory/'space.json'
    space.write_text(json.dumps({'training.optimizer.lr':[.001,.002]}))
    configuration=json.loads((directory/'traffic_state_pred/gru.json').read_text())
    configuration['runtime']['output_dir']=str(directory/'searches')
    configuration['runtime']['exp_id']='cli-study'
    configuration['data']['data_dir']=str(directory/'traffic_state_pred/data')
    configuration['data']['cache_dir']=str(directory/'traffic_state_pred/cache')
    config=directory/'search-config.json'
    config.write_text(json.dumps(configuration))
    result=run([root/'run_hyper.py','--config',config,'--space',space,'--max-trials','2'])
    study=json.loads(Path(result['artifacts']['search_report']).read_text())
    assert study['test_evaluations']==1 and len(study['trials'])==2
    report=dict(passed=True,python=platform.python_version(),torch=torch.__version__,cwd=str(directory),
        optional_packages_present=optional,tasks=tasks,tuning=dict(trials=2,test_evaluations=1),commands=records)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,tasks=list(tasks),commands=len(records),report=str(args.report))))


if __name__=='__main__':
    main()
