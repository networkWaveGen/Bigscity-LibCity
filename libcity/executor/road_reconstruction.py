"""Masked node reconstruction and explicit embedding export, outside forward."""
import json

import numpy as np
import torch

from libcity.executor.evaluate import ExperimentResult, StepResult, move_batch, write_metrics


class ReconstructionLoss:
    def __init__(self,bundle,config):
        self.bundle,self.name = bundle,config['name']

    def inputs(self,batch):
        return {name:batch[name] for name in ('node_features','node_indices')}

    def raw(self,prediction,target,raw_target=None):
        prediction = prediction.float() if prediction.dtype in (torch.float16,torch.bfloat16) else prediction
        target = target.float() if target.dtype in (torch.float16,torch.bfloat16) else target
        return self.bundle.scaler.inverse_transform(prediction), (
            raw_target if raw_target is not None else self.bundle.scaler.inverse_transform(target))

    def __call__(self,prediction,target,raw_target=None):
        prediction,target = self.raw(prediction,target,raw_target)
        if prediction.shape!=target.shape:
            raise ValueError('Road reconstruction prediction/target shapes differ')
        valid = torch.isfinite(target)
        count = int(valid.sum())
        if not count:
            return StepResult(prediction[valid].sum(),0)
        if not torch.isfinite(prediction[valid]).all():
            raise ValueError('Road reconstruction prediction is nonfinite on observed targets')
        difference = prediction[valid]-target[valid]
        objective = difference.square().mean() if self.name=='mse' else difference.abs().mean()
        return StepResult(objective,count)


def evaluate_reconstruction(model,loader,*,device,loss_fn,evaluation,directory=None,prefix=''):
    model.eval()
    total,count,absolute,squared = 0.,0,0.,0.
    outputs,truths,indices = [],[],[]
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch,device)
            prediction = model(loss_fn.inputs(batch))
            result = loss_fn(prediction,batch['y'],batch['y_raw'])
            raw,target = loss_fn.raw(prediction,batch['y'],batch['y_raw'])
            valid = torch.isfinite(target)
            difference = (raw.double()-target.double())[valid]
            absolute += difference.abs().sum().item()
            squared += difference.square().sum().item()
            count += result.weight
            total += result.loss.item()*result.weight
            if directory is not None and evaluation['save_predictions']:
                outputs.append(raw.cpu().numpy())
                truths.append(target.cpu().numpy())
                indices.append(batch['node_indices'].cpu().numpy())
        if not count:
            raise ValueError('Road reconstruction evaluation has no finite targets')
        values = dict(loss=total/count,MAE=absolute/count,MSE=squared/count,RMSE=(squared/count)**.5)
        metrics = {prefix+name:values[name] for name in dict.fromkeys([*evaluation['metrics'],'loss'])}
        artifacts = {}
        if directory is not None:
            artifacts = write_metrics(metrics,directory)
            path = directory/'embeddings.npy'
            features = loader.dataset.features.to(device)
            np.save(path,model.encode(features).cpu().numpy(),allow_pickle=False)
            artifacts['embeddings'] = path
            path = directory/'node_ids.json'
            path.write_text(json.dumps(list(loss_fn.bundle.metadata.info['node_ids']))+'\n',encoding='utf-8')
            artifacts['node_ids'] = path
            path = directory/'predictions.npz'
            if evaluation['save_predictions']:
                np.savez_compressed(path,predictions=np.concatenate(outputs),targets=np.concatenate(truths),
                                    node_indices=np.concatenate(indices))
                artifacts['predictions'] = path
            else:
                path.unlink(missing_ok=True)
    return ExperimentResult(metrics,artifacts)
