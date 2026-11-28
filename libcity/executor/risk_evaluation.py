"""Risk-region evaluation preserves unmodified raw prediction/truth exports."""
from pathlib import Path

import numpy as np
import torch

from libcity.evaluator.runtime_risk import RiskMetrics
from libcity.executor.evaluate import ExperimentResult,aggregate_loss,model_inputs,move_batch,write_metrics


def evaluate_risk(model,loader,*,device,loss_fn,evaluation,directory=None,prefix=''):
    model.eval()
    metrics = RiskMetrics(evaluation['metrics'],loss_fn.regions>0,evaluation.get('topk',10),
        evaluation['mask_value'],evaluation['mode'])
    total,weight = 0.,0
    predictions,targets = [],[]
    export = directory is not None and evaluation['save_predictions']
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch,device)
            prediction = model(model_inputs(batch,loss_fn.output_dim))
            step = loss_fn(prediction,batch['y'],batch.get('y_raw'))
            total += step.loss.item()*step.weight;weight += step.weight
            raw_prediction,raw_target = loss_fn.raw(prediction,batch['y'],batch.get('y_raw'))
            metrics.update(raw_prediction,raw_target)
            if export:
                predictions.append(raw_prediction.cpu().numpy());targets.append(raw_target.cpu().numpy())
    values = {prefix+name:value for name,value in metrics.compute().items()}
    values[prefix+'loss'] = aggregate_loss(loss_fn,total,weight)
    artifacts = {}
    if directory is not None:
        directory = Path(directory);artifacts = write_metrics(values,directory)
        path = directory/'predictions.npz'
        if export:
            np.savez_compressed(path,prediction=np.concatenate(predictions),truth=np.concatenate(targets),regions=(loss_fn.regions>0).numpy())
            artifacts['predictions'] = path
        else: path.unlink(missing_ok=True)
    return ExperimentResult(values,artifacts)
