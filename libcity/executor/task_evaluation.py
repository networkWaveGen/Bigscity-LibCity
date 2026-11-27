"""Task losses and streaming metrics plugged into the common native loop."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from libcity.executor.evaluate import ExperimentResult, StepResult, move_batch, write_metrics


class TaskLoss:
    def __init__(self,bundle,config):
        self.bundle,self.config=bundle,config
        self.output_dim=bundle.metadata.output_dim

    def inputs(self,batch):
        # Targets, local duration labels and identifiers never reach forward.
        return {key:value for key,value in batch.items() if key not in
                ('y','y_raw','local_y_raw','sample_id','traj_id')}


class RankingLoss(TaskLoss):
    def __call__(self,prediction,target,raw_target=None):
        return StepResult(F.nll_loss(prediction,target),target.numel())


class EdgeLoss(TaskLoss):
    def __call__(self,prediction,target,raw_target=None):
        return StepResult(-F.logsigmoid(prediction*target).mean(),target.numel())


class ETALoss(TaskLoss):
    def raw(self,prediction,target,raw_target=None):
        return self.bundle.scaler.inverse_transform(prediction.float()), (
            raw_target if raw_target is not None else self.bundle.scaler.inverse_transform(target.float()))

    def __call__(self,prediction,target,raw_target=None):
        prediction,target=self.raw(prediction,target,raw_target)
        residual=prediction-target
        value=residual.square().mean() if self.config['name']=='mse' else residual.abs().mean()
        return StepResult(value,target.numel())


def eta_step(loss_fn):
    def step(model,batch,state):
        prediction,local,valid=model.forward_with_aux(loss_fn.inputs(batch))
        main=loss_fn(prediction,batch['y'],batch['y_raw'])
        gap=model.kernel_size-1
        raw=batch['local_y_raw']
        target=(raw[:,gap:]-raw[:,:-gap])[valid].unsqueeze(-1)
        stats=loss_fn.bundle.metadata.info['statistics']['time_gap']
        local=local*(gap*stats[1])+gap*stats[0]
        auxiliary=((local-target).abs()/target.abs().clamp_min(loss_fn.config.get('eps',10.))).mean()
        alpha=loss_fn.config.get('alpha',.3)
        return StepResult((1-alpha)*main.loss+alpha*auxiliary,main.weight)
    return step


def evaluate_task(model,loader,*,device,loss_fn,evaluation,directory=None,prefix=''):
    model.eval()
    task=loss_fn.bundle.metadata.input_layout['kind']
    sums={name:0. for name in evaluation['metrics']}
    total,count=0.,0
    records=[]
    export=directory is not None and evaluation['save_predictions']
    squared=absolute=percentage=0.
    with torch.no_grad():
        for batch in loader:
            batch=move_batch(batch,device)
            prediction=model(loss_fn.inputs(batch))
            result=loss_fn(prediction,batch['y'],batch.get('y_raw'))
            if not torch.isfinite(result.loss) or not torch.isfinite(prediction).all():
                raise ValueError('Task prediction/loss must be finite')
            total+=result.loss.item()*result.weight
            count+=result.weight
            if task=='padded_trajectory':
                ranking=prediction.argsort(dim=1,descending=True,stable=True)
                rank=(ranking==batch['y'][:,None]).nonzero()[:,1]+1
                for name in sums:
                    metric,k=name.split('@')
                    valid=rank<=int(k)
                    values=valid.float() if metric=='Recall' else valid/rank if metric=='MRR' else valid/torch.log2(rank+1.)
                    sums[name]+=values.sum().item()
                if export:
                    ids=loss_fn.bundle.metadata.info['location_ids']
                    for identity,truth,order in zip(batch['sample_id'],batch['y'],ranking):
                        records.append(dict(sample_id=int(identity),truth=ids[int(truth)],
                            prediction=[ids[int(i)] for i in order[:max(evaluation['topk'])]]))
            elif task=='planned_route':
                raw,truth=loss_fn.raw(prediction,batch['y'],batch['y_raw'])
                difference=raw.double()-truth.double()
                absolute+=difference.abs().sum().item()
                squared+=difference.square().sum().item()
                percentage+=(difference.abs()/truth.abs().clamp_min(1e-8)).sum().item()
                if export:
                    for identity,trajectory,value,target in zip(batch['sample_id'],batch['traj_id'],raw,truth):
                        records.append(dict(sample_id=int(identity),traj_id=int(trajectory),prediction_seconds=float(value.item()),truth_seconds=float(target.item())))
            else:
                if 'accuracy' in sums:
                    sums['accuracy']+=((prediction>=0)==(batch['y']>0)).sum().item()
    if not count:
        raise ValueError('Evaluation loader has no samples')
    if task=='planned_route':
        values=dict(MAE=absolute/count,MSE=squared/count,RMSE=(squared/count)**.5,MAPE=percentage/count)
        metrics={name:values[name] for name in sums}
    else:
        metrics={name:(total if name=='loss' else value)/count for name,value in sums.items()}
    metrics['loss']=total/count
    metrics={prefix+key:float(value) for key,value in metrics.items()}
    artifacts={}
    if directory is not None:
        artifacts=write_metrics(metrics,directory)
        path=directory/'predictions.json'
        if export and task!='edge_negative_sampling':
            path.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
            artifacts['predictions']=path
        else:
            path.unlink(missing_ok=True)
        if task=='edge_negative_sampling':
            path=directory/'embeddings.npy'
            np.save(path,model.embeddings().cpu().numpy(),allow_pickle=False)
            artifacts['embeddings']=path
            path=directory/'node_ids.json'
            path.write_text(json.dumps(list(loss_fn.bundle.metadata.info['node_ids']))+'\n',encoding='utf-8')
            artifacts['node_ids']=path
    return ExperimentResult(metrics,artifacts)
