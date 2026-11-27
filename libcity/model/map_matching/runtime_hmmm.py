"""Direct algorithm entry: no optimizer, writer or model checkpoint is required."""
import json
import numpy as np
from libcity.model.map_matching.HMMM import HMMM
from libcity.evaluator.map_matching_evaluator import MapMatchingEvaluator
from libcity.executor.evaluate import ExperimentResult,write_metrics


def create_model(*,metadata,k=5,r=200,mu=0.,sigma_z=10.,beta=5.,mem_rate=1.):
    if k<1 or r<=0 or sigma_z<=0 or beta<=0 or not 0<=mem_rate<=1:
        raise ValueError('HMMM requires positive candidate/radius/noise parameters and mem_rate in [0,1]')
    return HMMM(dict(k=k,r=r,mu=mu,sigma_z=sigma_z,beta=beta,mem_rate=mem_rate),dict(metadata.info['features']))


def json_value(value):
    if isinstance(value,np.generic):
        return value.item()
    if hasattr(value,'isoformat'):
        return value.isoformat()
    raise TypeError(f'Unsupported prediction value {type(value).__name__}')


def evaluate(algorithm,bundle,evaluation,*,directory,logger):
    algorithm._logger=logger
    predictions=algorithm.run(bundle.test)
    evaluator=MapMatchingEvaluator(dict(metrics=evaluation['metrics']))
    evaluator._logger=logger
    evaluator.collect(dict(result=predictions,route=bundle.test['route'],rd_nwk=bundle.test['rd_nwk']))
    evaluator.evaluate()
    values=[result for groups in evaluator.evaluate_result.values() for result in groups.values()]
    metrics={name:float(np.mean([value[name] for value in values])) for name in evaluation['metrics']}
    artifacts=write_metrics(metrics,directory)
    path=directory/'metrics_by_trajectory.json'
    details={str(user):{str(trajectory):result for trajectory,result in groups.items()}
             for user,groups in evaluator.evaluate_result.items()}
    path.write_text(json.dumps(details,default=json_value,indent=2)+'\n',encoding='utf-8')
    artifacts['metrics_by_trajectory']=path
    path=directory/'predictions.json'
    if evaluation['save_predictions']:
        payload={str(user):{str(trajectory):rows.tolist() for trajectory,rows in groups.items()}
                 for user,groups in predictions.items()}
        path.write_text(json.dumps(payload,default=json_value,indent=2)+'\n',encoding='utf-8')
        artifacts['predictions']=path
    else:
        path.unlink(missing_ok=True)
    return ExperimentResult(metrics,artifacts)
