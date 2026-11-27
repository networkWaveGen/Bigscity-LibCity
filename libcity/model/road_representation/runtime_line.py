"""LINE's two objectives, without filesystem work in forward."""
from torch import nn
from libcity.model.road_representation.LINE import LINE_FIRST, LINE_SECOND


class LINE(nn.Module):
    def __init__(self,*,metadata,embedding_dim=32,order='second'):
        super().__init__()
        if order not in ('first','second') or embedding_dim<1:
            raise ValueError('LINE requires order first/second and positive embedding_dim')
        self.embed=(LINE_FIRST if order=='first' else LINE_SECOND)(metadata.num_nodes,embedding_dim)

    def forward(self,batch):
        return self.embed(batch['I'],batch['J'])

    def embeddings(self):
        return self.embed.node_emb.weight.detach()
