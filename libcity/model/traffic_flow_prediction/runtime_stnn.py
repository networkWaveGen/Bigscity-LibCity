"""Latent factor dynamics with explicit time/node axes and trainable lookup."""
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.traffic_flow_prediction.STNN import MLP


class STNN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,mode='refine',nhid=0,nlayers=1,
                 dropout_f=.1,dropout_d=.1,lookup='linear'):
        super().__init__()
        if mode not in (None,'refine','discover') or lookup not in ('linear','hard'):
            raise ValueError('STNN requires mode null/refine/discover and lookup linear/hard')
        if nlayers<1 or (nlayers>1 and nhid<1) or not 0<=dropout_f<1 or not 0<=dropout_d<1:
            raise ValueError('STNN requires positive hidden width for multiple layers and valid dropout')
        if metadata.input_layout.get('kind')!='window':
            raise ValueError('STNN requires contiguous input windows')
        self.window,self.horizon = input_window,output_window
        self.nodes,self.features,self.outputs = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.mode,self.lookup = mode,lookup
        identity = torch.eye(self.nodes)
        graph = torch.ones_like(identity) if mode=='discover' else torch.tensor(metadata.adjacency.copy(),dtype=torch.float32)
        relations = torch.stack((identity,graph),dim=1)
        if mode=='refine':
            relations = relations.ceil().clamp(0,1).bool()
        self.register_buffer('relations',relations)
        if mode=='refine':
            self.rel_weights = nn.Parameter(torch.full((int(relations[:,1].sum()),),.5))
        elif mode=='discover':
            self.rel_weights = nn.Parameter(torch.full((self.nodes,1,self.nodes),1./self.nodes))
        self.drop = nn.Dropout(dropout_f)
        self.factors = nn.Parameter(torch.empty(input_window,self.nodes,self.features))
        nn.init.uniform_(self.factors,-.1,.1)
        self.ffunc = nn.Linear(input_window*self.nodes*self.features,2*input_window*self.nodes)
        self.dynamic = MLP(self.features*2,nhid,self.features,nlayers,dropout_d)
        self.decoder = nn.Linear(self.features*input_window,output_window*self.outputs,bias=False)

    def get_relations(self):
        if self.mode is None:
            return self.relations
        weights = F.hardtanh(self.rel_weights,0,1)
        if self.mode=='refine':
            inter = weights.new_zeros(self.nodes,1,self.nodes).masked_scatter(self.relations[:,1:],weights)
        else:
            inter = weights
        return torch.cat((self.relations[:,:1].to(weights.dtype),inter),dim=1)

    def encode(self,source):
        coordinates = self.ffunc(source.flatten(1)).reshape(source.shape[0],self.window,self.nodes,2).sigmoid()
        time,node = coordinates[...,0]*(self.window-1),coordinates[...,1]*(self.nodes-1)
        if self.lookup=='hard':
            # Retained only for explicit numerical comparisons with old lookup.
            latent = self.factors[time.ceil().long(),node.ceil().long()]
        else:
            time0,node0 = time.floor().long(),node.floor().long()
            time1,node1 = (time0+1).clamp_max(self.window-1),(node0+1).clamp_max(self.nodes-1)
            time_weight,node_weight = (time-time0)[...,None],(node-node0)[...,None]
            latent = ((1-time_weight)*(1-node_weight)*self.factors[time0,node0]
                +(1-time_weight)*node_weight*self.factors[time0,node1]
                +time_weight*(1-node_weight)*self.factors[time1,node0]
                +time_weight*node_weight*self.factors[time1,node1])
        return self.drop(latent)

    def transition(self,latent):
        context = torch.einsum('nrm,btmf->btnrf',self.get_relations().to(latent.dtype),latent)
        return torch.tanh(self.dynamic(context.flatten(-2)))

    def forward(self,batch):
        latent = self.encode(batch['X'])
        for _ in range(self.horizon):
            latent = self.transition(latent)
        decoded = self.decoder(latent.transpose(1,2).flatten(2))
        return decoded.reshape(latent.shape[0],self.nodes,self.horizon,self.outputs).transpose(1,2)
