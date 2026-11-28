"""Risk-region squared weights and raw risk-bin weighting outside the model."""
import math

import numpy as np
import torch

from libcity.executor.evaluate import ForecastLoss,StepResult


def validate_bins(thresholds,weights):
    if (not isinstance(thresholds,list) or any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in thresholds)
            or any(right<=left for left,right in zip(thresholds,thresholds[1:]))):
        raise ValueError('Risk thresholds must be strictly increasing finite nonnegative values')
    if (not isinstance(weights,list) or len(weights)!=len(thresholds)+1 or
            any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in weights) or not any(weights)):
        raise ValueError('Risk weights must be nonnegative, nonempty and one longer than thresholds')


class RiskLoss(ForecastLoss):
    def __init__(self,bundle,config):
        super().__init__(config,bundle.scaler,1)
        if config['name']!='risk_mse' or config.get('mask_value') is not None:
            raise ValueError('RiskLoss requires risk_mse including real zero targets')
        thresholds = config.get('thresholds')
        weights = config.get('weights')
        thresholds = thresholds if thresholds is not None else bundle.metadata.info.get('risk_thresholds')
        weights = weights if weights is not None else bundle.metadata.info.get('risk_weights')
        validate_bins(list(thresholds) if isinstance(thresholds,tuple) else thresholds,
            list(weights) if isinstance(weights,tuple) else weights)
        self.thresholds,self.weights = torch.tensor(thresholds,dtype=torch.float64),torch.tensor(weights,dtype=torch.float64)
        region = np.array(bundle.metadata.static_data['risk_mask'],copy=True,dtype=np.float64)
        if region.shape!=(bundle.metadata.num_nodes,) or not np.isfinite(region).all() or np.any(region<0) or not np.any(region>0):
            raise ValueError('RiskLoss requires a finite nonnegative nonempty region mask')
        self.regions = torch.from_numpy(region)
        self.region_fraction = float(np.count_nonzero(region)/len(region))
        relative = region/region.max()
        self.squared_region_weights = torch.from_numpy(np.square(relative/relative.mean()))
        self._cache = {}

    def _constants(self,prediction):
        key = prediction.device,prediction.dtype
        if key not in self._cache:
            self._cache[key] = tuple(value.to(device=prediction.device,dtype=prediction.dtype) for value in (
                self.thresholds,self.weights,self.squared_region_weights))
        return self._cache[key]

    def __call__(self,prediction,target,raw_target=None):
        prediction,target = self.raw(prediction,target,raw_target)
        if prediction.shape!=target.shape or target.ndim!=4 or target.shape[-2:]!=(len(self.regions),1):
            raise ValueError('RiskLoss predictions/targets must share [B,H,N,1]')
        thresholds,weights,regions = self._constants(prediction)
        regions = regions[None,None,:,None].expand_as(target)
        valid = torch.isfinite(target)&(regions>0)
        count = int(valid.sum())
        if not count: return StepResult(prediction[valid].sum(),0)
        if not torch.isfinite(prediction[valid]).all() or torch.any(target[valid]<0):
            raise ValueError('RiskLoss requires finite predictions and nonnegative known risks')
        residual = prediction[valid]-target[valid]
        bins = torch.bucketize(target[valid].to(prediction.dtype),thresholds,right=False)
        # Fraction M/N retains the original full-grid mean when fully observed;
        # valid-region counts keep missing-value/tail accumulation additive.
        objective = (residual.square()*regions[valid]*weights[bins]).mean()*self.region_fraction
        return StepResult(objective,count)
