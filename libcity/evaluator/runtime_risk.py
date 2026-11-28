"""Per-frame risk rankings and global regression statistics over eligible cells."""
import torch

from libcity.evaluator.runtime_metrics import ForecastMetrics


RANK_METRICS = ('Precision','Recall','F1-Score','MAP','PCC')


class RiskMetrics:
    def __init__(self,names,regions,topk=10,mask_value=None,mode='single'):
        if type(topk) is not int or topk<1 or mode not in ('single','average'):
            raise ValueError('Risk metrics require positive topk and single/average mode')
        self.names,self.topk,self.mode = names,topk,mode
        self.regions = (torch.as_tensor(regions).detach().clone()>0).flatten().cpu()
        self.regression = ForecastMetrics([name for name in names if name not in RANK_METRICS],mask_value,mode)
        self.scores,self.frames = None,None

    def update(self,prediction,target):
        if prediction.shape!=target.shape or target.ndim!=4 or target.shape[-2:]!=(len(self.regions),1):
            raise ValueError('Risk metrics require matching [B,H,N,1]')
        prediction,target = prediction.detach().double(),target.detach().double()
        eligible = self.regions.to(target.device)[None,None,:,None]&torch.isfinite(target)
        if not torch.isfinite(prediction[eligible]).all():
            raise ValueError('Risk predictions must be finite on eligible known cells')
        self.regression.update(prediction,torch.where(eligible,target,float('nan')))
        batch,horizon,nodes,_ = target.shape
        valid = eligible[...,0].reshape(-1,nodes)
        truth = torch.where(valid,target[...,0].reshape(-1,nodes),0.)
        forecast = torch.where(valid,prediction[...,0].reshape(-1,nodes),0.)
        order = prediction[...,0].reshape(-1,nodes).masked_fill(~valid,float('-inf')).argsort(dim=-1,descending=True,stable=True)
        k = min(self.topk,nodes)
        hits = ((truth>0)&valid).gather(-1,order[:,:k]).double()
        selected = valid.sum(-1).clamp_max(self.topk)
        positives = ((truth>0)&valid).sum(-1)
        matched = hits.sum(-1)
        precision = matched/selected.clamp_min(1)
        recall = matched/positives.clamp_min(1)
        f1 = 2*precision*recall/(precision+recall).clamp_min(torch.finfo(torch.float64).tiny)
        average_precision = (hits.cumsum(-1)/torch.arange(1,k+1,device=truth.device)*hits).sum(-1)/torch.minimum(positives,selected).clamp_min(1)
        count = valid.sum(-1).clamp_min(1)
        centered_truth = torch.where(valid,truth-truth.sum(-1,keepdim=True)/count[:,None],0.)
        centered_forecast = torch.where(valid,forecast-forecast.sum(-1,keepdim=True)/count[:,None],0.)
        covariance = (centered_truth*centered_forecast).sum(-1)
        truth_variance = centered_truth.square().sum(-1)
        forecast_variance = centered_forecast.square().sum(-1)
        denominator = (truth_variance*forecast_variance).sqrt()
        pcc = torch.where(denominator>0,covariance/denominator.clamp_min(torch.finfo(torch.float64).tiny),0.).clamp(-1,1)
        values = torch.stack((precision,recall,f1,average_precision,pcc),-1).reshape(batch,horizon,-1).sum(0).cpu()
        if self.scores is None:
            self.scores = torch.zeros_like(values);self.frames = torch.zeros(horizon,dtype=torch.float64)
        self.scores += values;self.frames += batch

    def compute(self):
        result = self.regression.compute()
        for horizon in range(len(self.scores)):
            selection = slice(0,horizon+1) if self.mode=='average' else slice(horizon,horizon+1)
            scores = self.scores[selection].sum(0)/self.frames[selection].sum()
            for name in self.names:
                if name in RANK_METRICS: result[f'{name}@{horizon+1}'] = float(scores[RANK_METRICS.index(name)])
        return result
