"""Backend-neutral finite search using the same experiment entry as normal runs.

Trial fitting never evaluates the test loader. Only the validation winner receives
one independent test evaluation. Grid/random use Python's standard library, leaving
the decision about Optuna/Ray migration for the plan's deferred discussion.
"""
from copy import deepcopy
from itertools import islice, product
import json
import math
from pathlib import Path
import random
import uuid

from libcity.config.runtime import load_config, merge_config
from libcity.executor.evaluate import ExperimentResult
from libcity.model.runtime_registry import model_spec
from libcity.pipeline.experiment import run_experiment


def configuration_patch(parameters):
    result={}
    for name,value in parameters.items():
        parts=name.split('.')
        if len(parts)<2 or any(not item for item in parts):
            raise ValueError('Search keys must be dotted configuration keys')
        if name.startswith('runtime.') or name in ('model.name','data.dataset','training.monitor','training.direction'):
            raise ValueError('Search model/task/dataset identity and runtime cannot change within a study')
        current=result
        for part in parts[:-1]:
            if part in current and not isinstance(current[part],dict):
                raise ValueError('Search space has conflicting parent/child keys')
            current=current.setdefault(part,{})
        if parts[-1] in current:
            raise ValueError('Search space has conflicting parent/child keys')
        current[parts[-1]]=deepcopy(value)
    return result


def candidates(space,algorithm,max_trials,seed):
    if not isinstance(space,dict) or not space:
        raise ValueError('Search space must be a nonempty JSON object')
    if type(max_trials) is not int or max_trials<1:
        raise ValueError('max_trials must be a positive integer')
    for key,values in space.items():
        if not isinstance(values,list) or not values:
            raise ValueError(f'Search domain {key} must be a nonempty list')
    keys=list(space)
    domains=list(space.values())
    if algorithm=='grid':
        combinations=islice(product(*domains),max_trials)
    elif algorithm=='random':
        total=math.prod(map(len,domains))
        def decode(index):
            result=[]
            for values in reversed(domains):
                index,position=divmod(index,len(values))
                result.append(values[position])
            return tuple(reversed(result))
        combinations=(decode(index) for index in random.Random(seed).sample(range(total),min(total,max_trials)))
    else:
        raise ValueError('Search algorithm must be grid or random')
    return [dict(zip(keys,values)) for values in combinations]


def run_search(config_file=None,space_file=None,*,config=None,space=None,algorithm='grid',max_trials=10,overrides=None):
    if space_file is not None and space is not None:
        raise ValueError('Supply space_file or space, not both')
    if space_file is not None:
        space=json.loads(Path(space_file).read_text(encoding='utf-8'))
    base=load_config(config_file,overrides,config=config)
    spec=model_spec(base['model']['name'])
    if not spec.supports_validation or not spec.requires_checkpoint:
        raise ValueError('Tuning requires validation scores and a trained model checkpoint')
    trials=candidates(space,algorithm,max_trials,base['runtime']['seed'])
    # Resolve every candidate before creating outputs or training anything.
    configs=[load_config(config=merge_config(base,configuration_patch(params))) for params in trials]
    study=Path(base['runtime']['output_dir'])/(base['runtime']['exp_id'] or 'search-'+uuid.uuid4().hex[:12])
    study.mkdir(parents=True,exist_ok=False)
    report=dict(algorithm=algorithm,seed=base['runtime']['seed'],direction=base['training']['direction'],
                monitor=base['training']['monitor'],test_evaluations=0,trials=[])
    report_file=study/'search.json'
    def save():
        report_file.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    best=None
    for index,(parameters,cfg) in enumerate(zip(trials,configs)):
        cfg['runtime'].update(output_dir=str(study/'trials'),exp_id=f'trial-{index:03d}')
        try:
            result=run_experiment('train',config=cfg,evaluate_test=False)
        except Exception as error:
            report['trials'].append(dict(index=index,parameters=parameters,status='failed',error=f'{type(error).__name__}: {error}'))
            save()
            raise
        score=float(result.metrics['best_validation'])
        if not math.isfinite(score):
            raise ValueError('Trial returned a non-finite validation score')
        record=dict(index=index,parameters=parameters,status='complete',validation_score=score,
                    resolved_config=str(result.artifacts['resolved_config']),checkpoint=str(result.artifacts['best']))
        report['trials'].append(record)
        better=best is None or (score<best[0] if report['direction']=='min' else score>best[0])
        if better:
            best=(score,index,cfg,result.artifacts['best'])
        save()
    score,index,cfg,checkpoint=best
    report.update(best_trial=index,best_validation=score)
    selected=deepcopy(cfg)
    selected['runtime'].update(output_dir=str(study),exp_id='selected-test')
    result=run_experiment('evaluate',config=selected,checkpoint=checkpoint)
    report.update(test_evaluations=1,test_metrics=result.metrics)
    save()
    result.metrics['best_validation']=score
    result.artifacts.update(search_report=report_file,best=Path(checkpoint))
    return ExperimentResult(result.metrics,result.artifacts)
