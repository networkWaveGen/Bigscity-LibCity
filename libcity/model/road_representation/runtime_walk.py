"""Safe tensor-only node vectors fitted by an explicitly external optimizer."""
import json
import math

import torch
from torch import nn


def encoded(value):
    return torch.tensor(list(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')),dtype=torch.uint8)


class WalkEmbeddings(nn.Module):
    def __init__(self,*,metadata,algorithm,output_dim=128,num_walks=10,walk_length=80,p=1.,q=1.,restart=0.):
        super().__init__()
        if min(output_dim,num_walks,walk_length)<1 or not all(math.isfinite(v) and v>0 for v in (p,q)) or not 0<=restart<=1:
            raise ValueError('Walk embeddings require positive dimensions/walks/length/p/q and restart in [0,1]')
        self.algorithm,self.num_walks,self.walk_length = algorithm,num_walks,walk_length
        self.p,self.q,self.restart = float(p),float(q),float(restart)
        info = metadata.info
        self.register_buffer('vectors',torch.zeros(metadata.num_nodes,output_dim))
        self.register_buffer('fitted',torch.tensor(False))
        self.register_buffer('node_ids_utf8',encoded(list(info['node_ids'])))
        self.register_buffer('algorithm_utf8',encoded(dict(algorithm=algorithm,num_walks=num_walks,walk_length=walk_length,
            p=self.p,q=self.q,restart=self.restart,directed=info['directed'],graph_kind=info['graph_kind'])))
        self.register_buffer('graph_edges',torch.tensor(info['edge_index'].copy()))
        self.register_buffer('graph_weights',torch.tensor(info['edge_weights'].copy()))

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if set(state_dict)!=set(super().state_dict()) or not isinstance(state_dict['fitted'],torch.Tensor) or state_dict['fitted'].dtype!=torch.bool or state_dict['fitted'].numel()!=1 or not bool(state_dict['fitted'].item()):
            raise ValueError('Walk evaluation requires complete fitted tensor-only embedding weights')
        if state_dict['vectors'].shape!=self.vectors.shape or not state_dict['vectors'].is_floating_point() or not torch.isfinite(state_dict['vectors']).all():
            raise ValueError('Walk embedding shape/values do not match this model')
        for name in ('node_ids_utf8','algorithm_utf8','graph_edges','graph_weights'):
            if not torch.equal(state_dict[name].cpu(),getattr(self,name).cpu()):
                raise ValueError(f'Walk checkpoint node identities, algorithm or graph changed: {name}')
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def embeddings(self):
        if not bool(self.fitted) or not torch.isfinite(self.vectors).all():
            raise ValueError('Walk embeddings have not been fitted or contain nonfinite values')
        return self.vectors

    def forward(self,batch):
        indices = batch['node_indices']
        if (indices<0).any() or (indices>=len(self.vectors)).any():
            raise ValueError('Walk node indices are outside the vocabulary')
        return self.embeddings()[indices]


def create_node2vec(*,metadata,output_dim=128,num_walks=10,walk_length=80,p=2.,q=1.):
    return WalkEmbeddings(metadata=metadata,algorithm='node2vec',output_dim=output_dim,num_walks=num_walks,walk_length=walk_length,p=p,q=q)


def create_deepwalk(*,metadata,output_dim=128,num_walks=10,walk_length=80,restart=0.):
    return WalkEmbeddings(metadata=metadata,algorithm='deepwalk',output_dim=output_dim,num_walks=num_walks,walk_length=walk_length,restart=restart)
