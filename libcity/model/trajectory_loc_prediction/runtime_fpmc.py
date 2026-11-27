"""Factorized personalized Markov scores with valid full-vocabulary probabilities."""
import torch
from torch import nn
from torch.nn import functional as F


class FPMC(nn.Module):
    def __init__(self, *, metadata, embedding_size=64):
        super().__init__()
        if embedding_size < 1 or 'uid_size' not in metadata.info:
            raise ValueError('FPMC requires positive embedding_size and personalized trajectory metadata')
        locations, padding = metadata.info['loc_size'], metadata.info['loc_pad']
        self.UI_emb = nn.Embedding(metadata.info['uid_size'], embedding_size)
        self.IU_emb = nn.Embedding(locations, embedding_size, padding_idx=padding)
        self.LI_emb = nn.Embedding(locations, embedding_size, padding_idx=padding)
        self.IL_emb = nn.Embedding(locations, embedding_size, padding_idx=padding)
        for module in (self.UI_emb, self.IU_emb, self.LI_emb, self.IL_emb):
            nn.init.xavier_normal_(module.weight)

    def forward(self, batch):
        last = batch['current_loc'].gather(1, (batch['lengths']-1).unsqueeze(1)).squeeze(1)
        preference = self.UI_emb(batch['uid']) @ self.IU_emb.weight[1:].T
        transition = self.LI_emb(last) @ self.IL_emb.weight[1:].T
        # Padding ID 0 is never a location target; the task target uses zero-based IDs.
        return F.log_softmax(preference+transition, dim=-1)
