"""Global per-horizon statistics; final metrics do not depend on batch boundaries."""
import math
import torch


class ForecastMetrics:
    def __init__(self, names, mask_value=0., mode='single'):
        self.names, self.mask_value, self.mode = names, mask_value, mode
        self.statistics = None

    def update(self, prediction, target):
        if prediction.shape != target.shape or target.ndim != 4:
            raise ValueError('Forecast predictions and targets must share shape (B,T,N,D)')
        prediction, target = prediction.detach().double(), target.detach().double()
        valid = torch.isfinite(target)
        if self.mask_value is not None:
            valid &= target != self.mask_value
        if not torch.isfinite(prediction[valid]).all():
            raise ValueError('Predictions contain non-finite values on valid targets')
        delta = torch.where(valid, prediction - target, 0.)
        truth = torch.where(valid, target, 0.)
        percentage_valid = valid & (target != 0)
        ratio = torch.where(percentage_valid, delta.abs() / target.abs(), 0.)
        dimensions = (0, 2, 3)
        values = torch.stack([valid.sum(dimensions), delta.abs().sum(dimensions), delta.square().sum(dimensions),
            ratio.sum(dimensions), percentage_valid.sum(dimensions), truth.sum(dimensions), truth.square().sum(dimensions),
            delta.sum(dimensions)], dim=1).cpu()
        if self.statistics is None:
            self.statistics = torch.zeros_like(values)
        self.statistics += values

    def compute(self):
        if self.statistics is None:
            raise ValueError('Evaluation data must contain at least one batch')
        result = {}
        for index in range(len(self.statistics)):
            stats = self.statistics[:index + 1].sum(0) if self.mode == 'average' else self.statistics[index]
            n, absolute, squared, percentage, n_percentage, truth, truth2, delta = stats.tolist()
            variance = truth2 - truth * truth / n if n else 0.
            for requested in self.names:
                name = requested.removeprefix('masked_')
                if not n:
                    value = 0.
                elif name == 'MAE':
                    value = absolute / n
                elif name == 'MSE':
                    value = squared / n
                elif name == 'RMSE':
                    value = math.sqrt(squared / n)
                elif name == 'MAPE':
                    value = percentage / n_percentage if n_percentage else 0.
                elif name == 'R2':
                    value = 1 - squared / variance if variance > 0 else float(squared == 0)
                else:
                    residual_variance = squared - delta * delta / n
                    value = 1 - residual_variance / variance if variance > 0 else float(residual_variance == 0)
                result[f'{requested}@{index + 1}'] = float(value)
        return result
