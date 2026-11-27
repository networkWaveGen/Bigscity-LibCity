"""Device movement, task loss and inference-only evaluation."""
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import torch

from libcity.evaluator.runtime_metrics import ForecastMetrics


@dataclass
class ExperimentResult:
    metrics: dict[str, float]
    artifacts: dict[str, Path] = field(default_factory=dict)


@dataclass
class StepResult:
    loss: torch.Tensor
    weight: int


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def model_inputs(batch, output_dim):
    inputs = {'X': batch['X']}
    if batch['X'].shape[-1] > output_dim:
        # Transitional legacy extra channels; feature semantics move in stage 3.
        inputs['future_features'] = batch['y'][..., output_dim:]
    return inputs


class ForecastLoss:
    def __init__(self, config, scaler, output_dim):
        self.name, self.mask_value = config['name'], config.get('mask_value', 0.)
        self.scaler, self.output_dim = scaler, output_dim

    def raw(self, prediction, target):
        target = target[..., :self.output_dim]
        return self.scaler.inverse_transform(prediction.float()), self.scaler.inverse_transform(target.float())

    def __call__(self, prediction, target):
        prediction, target = self.raw(prediction, target)
        if prediction.shape != target.shape:
            raise ValueError(f'Loss shape mismatch: prediction {prediction.shape}, target {target.shape}')
        valid = torch.isfinite(target)
        if self.name.startswith('masked_') and self.mask_value is not None:
            valid &= target != self.mask_value
        count = int(valid.sum().item())
        if not count:
            return StepResult(prediction[valid].sum(), 0)
        if not torch.isfinite(prediction[valid]).all():
            raise ValueError('Training/evaluation prediction contains non-finite values')
        residual = prediction[valid] - target[valid]
        loss = residual.square().mean() if self.name.endswith('mse') else residual.abs().mean()
        return StepResult(loss, count)


def supervised_step(loss_fn):
    def step(model, batch, state):
        return loss_fn(model(model_inputs(batch, loss_fn.output_dim)), batch['y'])
    return step


def write_metrics(metrics,directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'metrics.json'
    path.write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    artifacts={'metrics_json':path}
    path=directory/'metrics.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(('metric','value'))
        writer.writerows(metrics.items())
    artifacts['metrics_csv']=path
    return artifacts


def evaluate_loop(model, loader, *, device, loss_fn, evaluation, directory=None, prefix=''):
    model.eval()
    accumulator = ForecastMetrics(evaluation['metrics'], evaluation['mask_value'], evaluation['mode'])
    total, weight = 0., 0
    export = directory is not None and evaluation['save_predictions']
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            prediction = model(model_inputs(batch, loss_fn.output_dim))
            step = loss_fn(prediction, batch['y'])
            total += step.loss.item() * step.weight
            weight += step.weight
            raw_prediction, raw_target = loss_fn.raw(prediction, batch['y'])
            accumulator.update(raw_prediction, raw_target)
            if export:
                predictions.append(raw_prediction.cpu().numpy())
                targets.append(raw_target.cpu().numpy())
    metrics = {prefix + key: value for key, value in accumulator.compute().items()}
    metrics[prefix + 'loss'] = total / weight if weight else 0.
    artifacts = {}
    if directory is not None:
        directory = Path(directory)
        artifacts=write_metrics(metrics,directory)
        path = directory / 'predictions.npz'
        if export:
            np.savez_compressed(path, prediction=np.concatenate(predictions), truth=np.concatenate(targets))
            artifacts['predictions'] = path
        else:
            path.unlink(missing_ok=True)
    return ExperimentResult(metrics, artifacts)
