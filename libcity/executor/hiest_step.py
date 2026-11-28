"""Prediction plus per-sample stable hierarchy reconstruction and cosine costs."""
import numpy as np
import torch
from torch.nn import functional as F

from libcity.executor.evaluate import ForecastLoss,StepResult
from libcity.model.traffic_speed_prediction.runtime_hiest import feature_gram


class HIESTLoss(ForecastLoss):
    def __init__(self,bundle,config):
        super().__init__(config,bundle.scaler,bundle.metadata.output_dim)
        self.weights=tuple(config[key] for key in ('original_weight','regional_weight','orthogonal_weight'))
        self.membership=torch.tensor(np.array(bundle.metadata.static_data['region_membership'],copy=True))
        self.original=torch.tensor(np.array(bundle.metadata.static_data['original_reconstruction'],copy=True))
        self.regional=torch.tensor(np.array(bundle.metadata.static_data['regional_reconstruction'],copy=True))
        self.cache={}

    def constants(self,source):
        key=source.device,source.dtype
        if key not in self.cache: self.cache[key]=tuple(value.to(device=source.device,dtype=source.dtype) for value in (self.membership,self.original,self.regional))
        return self.cache[key]


def auxiliary_costs(regional,global_,mapping,membership,original_target,regional_target):
    dtype=torch.float32 if regional.dtype in (torch.float16,torch.bfloat16) else regional.dtype
    with torch.autocast(regional.device.type,enabled=False):
        regional,global_,mapping=regional.to(dtype),global_.to(dtype),mapping.to(dtype)
        membership=membership.to(device=regional.device,dtype=dtype)
        original=torch.einsum('nr,bcrt->bcnt',membership,regional)
        reconstructed=torch.einsum('rg,bcgt->bcrt',mapping,global_)
        first=feature_gram(original);second=feature_gram(reconstructed)
        original_loss=F.binary_cross_entropy_with_logits(first,original_target.to(first).expand_as(first),reduction='none').mean((-1,-2))
        regional_loss=F.binary_cross_entropy_with_logits(second,regional_target.to(second).expand_as(second),reduction='none').mean((-1,-2))
        vectors=F.normalize(global_.transpose(1,2).flatten(2),dim=-1,eps=1e-6)
        similarities=vectors@vectors.transpose(-1,-2)
        count=global_.shape[2]
        orthogonal=(similarities[:,torch.triu(torch.ones(count,count,device=global_.device,dtype=torch.bool),diagonal=1)].mean(-1)
            if count>1 else global_.flatten(1).sum(-1)*0.)
    return original_loss,regional_loss,orthogonal


def auxiliary_step(loss_fn):
    def step(model,batch,state):
        prediction,regional,global_,mapping=model.forward_with_aux(loss_fn.inputs(batch))
        main=loss_fn(prediction,batch['y'],batch.get('y_raw'))
        if not main.weight: return main
        _,target=loss_fn.raw(prediction,batch['y'],batch.get('y_raw'))
        valid=torch.isfinite(target)
        if loss_fn.name.startswith('masked_') and loss_fn.mask_value is not None: valid&=target!=loss_fn.mask_value
        dtype=torch.float32 if regional.dtype in (torch.float16,torch.bfloat16) else regional.dtype
        counts=valid.flatten(1).sum(-1).to(dtype)
        membership,original_target,regional_target=loss_fn.constants(regional.to(dtype))
        costs=auxiliary_costs(regional,global_,mapping,membership,original_target,regional_target)
        auxiliary=sum(weight*cost for weight,cost in zip(loss_fn.weights,costs))
        return StepResult(main.loss+(auxiliary*counts).sum()/main.weight,main.weight)
    return step
