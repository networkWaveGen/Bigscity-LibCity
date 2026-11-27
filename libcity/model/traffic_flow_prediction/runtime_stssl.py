"""STSSL inference is deterministic; view augmentation and auxiliary heads are training-only."""
import torch
from torch import nn
from torch.nn import functional as F
from libcity.model.traffic_flow_prediction.STSSL import STEncoder,MLP,TemporalHeteroModel,SpatialHeteroModel


def augment_graph(similarity,graph,percent):
    graph=graph.clone()
    edges=torch.tril(graph>0,diagonal=-1).nonzero()
    count=min(len(edges),int(len(edges)*percent/2))
    if not count:
        return graph
    scores=similarity[edges[:,0],edges[:,1]]
    probabilities=(1-torch.softmax(scores,0)).clamp_min(1e-8)
    selected=edges[torch.multinomial(probabilities,count,replacement=False)]
    graph[selected[:,0],selected[:,1]]=0
    graph[selected[:,1],selected[:,0]]=0
    candidates=torch.tril(graph==0,diagonal=-1).nonzero()
    if len(candidates):
        probabilities=torch.softmax(similarity[candidates[:,0],candidates[:,1]],0)
        selected=candidates[torch.multinomial(probabilities,min(count,len(candidates)),replacement=False)]
        graph[selected[:,0],selected[:,1]]=1
        graph[selected[:,1],selected[:,0]]=1
    return graph


class STSSL(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,d_model=64,dropout=.2,nmb_prototype=6,shm_temp=.5,percent=.1):
        super().__init__()
        if input_window<9 or output_window!=1 or d_model<4 or d_model%2 or not 0<=percent<1:
            raise ValueError('STSSL needs history>=9, output_window=1, even d_model>=4 and percent in [0,1)')
        self.percent=percent
        self.encoder=STEncoder(Kt=3,Ks=3,blocks=[[metadata.feature_dim,d_model//2,d_model],[d_model,d_model//2,d_model]],
            input_length=input_window,num_nodes=metadata.num_nodes,droprate=dropout)
        self.mlp=MLP(d_model,metadata.output_dim)
        self.thm=TemporalHeteroModel(d_model,1,metadata.num_nodes,None)
        self.shm=SpatialHeteroModel(d_model,nmb_prototype,1,shm_temp)
        graph=torch.tensor(metadata.adjacency.copy(),dtype=torch.float32)
        graph=torch.maximum(graph,graph.T)
        graph.fill_diagonal_(0.)
        self.register_buffer('graph',graph)

    def forward(self,batch):
        return self.mlp(self.encoder(batch['X'],self.graph))

    def forward_with_aux(self,batch):
        source=batch['X']
        first=self.encoder(source,self.graph)
        # The old cosine helper divides by zero for empty-flow regions.
        # Normalize with epsilon before using similarities as probabilities.
        spatial=torch.nan_to_num(self.encoder.s_sim_mx.detach())
        temporal=torch.nan_to_num(self.encoder.t_sim_mx.detach()).permute(1,0,2)
        graph=augment_graph(spatial,self.graph,self.percent*2)
        view=source.clone()
        count=min(view.shape[0]*view.shape[1]*view.shape[2],int(temporal.numel()*self.percent))
        if count:
            probabilities=(1-temporal.reshape(-1)).clamp_min(1e-8)
            indices=torch.multinomial(probabilities,count,replacement=False)
            view.reshape(-1,view.shape[-1])[indices]=0
        second=self.encoder(view,graph)
        return self.mlp(first),self.thm(first,second),self.shm(first,second)
