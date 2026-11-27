"""Bounded geographic negatives and importance-weighted binary objectives."""
from collections import OrderedDict

import numpy as np
from sklearn.neighbors import BallTree
import torch
from torch.nn import functional as F

from libcity.executor.evaluate import StepResult
from libcity.executor.task_evaluation import TaskLoss


class GeographicSampler:
    def __init__(self,coordinates,nearest,visited=(),exclude_visited=False):
        self.coordinates = np.radians(np.asarray(coordinates)[:,::-1])
        self.tree = BallTree(self.coordinates,metric='haversine')
        self.nearest = min(nearest,len(coordinates)-1)
        self.visited,self.exclude_visited = visited,exclude_visited
        self.cache = OrderedDict()

    def neighbours(self,positive):
        if positive in self.cache:
            self.cache.move_to_end(positive)
            return self.cache[positive]
        if self.nearest==0: return np.empty(0,dtype=np.int64)
        distances,indices = self.tree.query(self.coordinates[positive:positive+1],k=min(self.nearest+1,len(self.coordinates)))
        order = np.lexsort((indices[0],distances[0]))
        values = indices[0,order]
        values = values[values!=positive][:self.nearest].copy()
        values.setflags(write=False)
        self.cache[positive] = values
        if len(self.cache)>128: self.cache.popitem(last=False)
        return values

    def sample(self,targets,users,count):
        pools = []
        for positive,user in zip(targets.detach().cpu().tolist(),users.detach().cpu().tolist()):
            pool = self.neighbours(positive)
            if self.exclude_visited and len(pool):
                filtered = pool[~np.isin(pool,self.visited[user])]
                # Exhaustion falls back to geographic nonpositives, without retries.
                if len(filtered): pool = filtered
            pools.append(torch.tensor(pool,dtype=torch.long,device=targets.device))
        if not self.nearest:
            return targets.new_empty((len(targets),0)),torch.empty(len(targets),0,device=targets.device)
        negatives = torch.stack([pool[torch.randint(len(pool),(count,),device=targets.device)] for pool in pools])
        probabilities = torch.tensor([1/len(pool) for pool in pools],device=targets.device)[:,None].expand(-1,count)
        return negatives,probabilities


class GeoSANLoss(TaskLoss):
    def __init__(self,bundle,config):
        super().__init__(bundle,config)
        self.temperature,self.count = config['temperature'],config['negative_samples']
        self._sampler = None

    @property
    def sampler(self):
        # Full-vocabulary evaluation needs the loss, not a training KNN index.
        if self._sampler is None:
            info = self.bundle.metadata.info
            self._sampler = GeographicSampler(info['coordinates'],self.config['num_nearest'],
                info['training_visited'],self.config['exclude_visited'])
        return self._sampler

    def selected(self,scores,probabilities):
        scores = scores.float() if scores.dtype in (torch.float16,torch.bfloat16) else scores
        positive,negative = scores[:,0],scores[:,1:]
        loss = F.softplus(-positive)
        if negative.shape[1]:
            weights = (negative/self.temperature-probabilities.to(scores.dtype).log()).softmax(-1)
            loss = loss+(F.softplus(negative)*weights).sum(-1)
        return StepResult(loss.mean(),len(scores))

    def __call__(self,scores,target,raw_target=None):
        scores = scores.float() if scores.dtype in (torch.float16,torch.bfloat16) else scores
        positive = scores.gather(1,target[:,None]).squeeze(1)
        loss = F.softplus(-positive)
        if scores.shape[1]>1:
            valid = torch.ones_like(scores,dtype=torch.bool).scatter(1,target[:,None],False)
            weights = (scores/self.temperature).masked_fill(~valid,-torch.inf).softmax(-1)
            loss = loss+(F.softplus(scores)*weights).sum(-1)
        return StepResult(loss.mean(),len(scores))


def sampled_step(loss_fn):
    def step(model,batch,state):
        negatives,probabilities = loss_fn.sampler.sample(batch['y'],batch['uid'],loss_fn.count)
        candidates = torch.cat((batch['y'][:,None],negatives),dim=-1)
        scores = model.forward_candidates(loss_fn.inputs(batch),candidates)
        return loss_fn.selected(scores,probabilities)
    return step
