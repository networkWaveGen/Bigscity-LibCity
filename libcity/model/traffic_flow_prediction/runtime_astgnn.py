"""ASTGNN layers with explicit CPT layout, graph buffers and training-only teacher inputs."""
from copy import deepcopy
import torch
from torch import nn

from libcity.model.traffic_flow_prediction import ASTGNN as layers


class TemporalEncoding(layers.TemporalPositionalEncoding):
    def forward(self,value):
        position = self.pe[:,:,self.lookup_index,:] if self.lookup_index is not None else self.pe[:,:,:value.size(2),:]
        # Positional encoding must not detach the learned input projection.
        return self.dropout(value+position)


class ASTGNN(nn.Module):
    def __init__(self, *,metadata,input_window,output_window,num_layers=3,d_model=64,nb_head=8,dropout=0.,
                 aware_temporal_context=True,ScaledSAt=True,SE=True,TE=True,kernel_size=1,smooth_layer_num=1):
        super().__init__()
        layout = metadata.input_layout
        if layout.get('kind')!='cpt':
            raise ValueError('ASTGNN requires data.sampling.name=cpt')
        self.output_dim,self.output_window = metadata.output_dim,output_window
        adjacency = torch.tensor(metadata.adjacency.copy(),dtype=torch.float32)
        adjacency += torch.eye(metadata.num_nodes)
        adjacency = adjacency/adjacency.sum(1,keepdim=True)
        src_dense,trg_dense = nn.Linear(metadata.feature_dim,d_model),nn.Linear(self.output_dim,d_model)
        graph_layer = layers.spatialAttentionScaledGCN if ScaledSAt else layers.spatialGCN
        graph_ff = layers.PositionWiseGCNFeedForward(graph_layer(adjacency,d_model,d_model),dropout=dropout)
        sampling = layout['sampling']
        if aware_temporal_context:
            args = (nb_head,d_model,sampling.get('trend',0),sampling.get('period',0),sampling.get('closeness',0),output_window,kernel_size)
            ss = layers.MultiHeadAttentionAwareTemporalContex_q1d_k1d(*args,dropout=dropout)
            st = layers.MultiHeadAttentionAwareTemporalContex_qc_k1d(*args,dropout=dropout)
            tt = layers.MultiHeadAttentionAwareTemporalContex_qc_kc(*args,dropout=dropout)
        else:
            ss = layers.MultiHeadAttention(nb_head,d_model,dropout=dropout)
            st,tt = deepcopy(ss),deepcopy(ss)
        encoder_embedding,decoder_embedding = [src_dense],[trg_dense]
        if TE:
            encoder_embedding.append(TemporalEncoding(d_model,dropout,layout['max_position'],list(layout['temporal_indices'])))
            decoder_embedding.append(TemporalEncoding(d_model,dropout,output_window))
        if SE:
            spatial = layers.SpatialPositionalEncoding(d_model,metadata.num_nodes,dropout,
                layers.GCN(adjacency,d_model,d_model),smooth_layer_num=smooth_layer_num)
            encoder_embedding.append(deepcopy(spatial))
            decoder_embedding.append(deepcopy(spatial))
        self.encoder = layers.Encoder(layers.EncoderLayer(d_model,ss,deepcopy(graph_ff),dropout,
            residual_connection=True,use_LayerNorm=True),num_layers)
        self.decoder = layers.Decoder(layers.DecoderLayer(d_model,tt,st,deepcopy(graph_ff),dropout,
            residual_connection=True,use_LayerNorm=True),num_layers)
        self.src_embed,self.trg_embed = nn.Sequential(*encoder_embedding),nn.Sequential(*decoder_embedding)
        self.prediction_generator = nn.Linear(d_model,self.output_dim)
        for module in self.modules():
            if 'sym_norm_Adj_matrix' in module.__dict__:
                graph = module.__dict__.pop('sym_norm_Adj_matrix')
                module.register_buffer('sym_norm_Adj_matrix',graph)
        for parameter in self.parameters():
            if parameter.dim()>1:
                nn.init.xavier_uniform_(parameter)

    def encode(self,history):
        return self.encoder(self.src_embed(history))

    def decode(self,inputs,encoded):
        return self.prediction_generator(self.decoder(self.trg_embed(inputs),encoded))

    def forward_with_teacher(self,batch,targets):
        history = batch['X'].transpose(1,2)
        targets = targets[...,:self.output_dim].transpose(1,2)
        decoder = torch.cat((history[:,:,-1:,:self.output_dim],targets[:,:,:-1]),dim=2)
        return self.decode(decoder,self.encode(history)).transpose(1,2)

    def forward(self,batch):
        history = batch['X'].transpose(1,2)
        encoded = self.encode(history)
        start = history[:,:,-1:,:self.output_dim]
        decoder = start
        for _ in range(self.output_window):
            prediction = self.decode(decoder,encoded)
            decoder = torch.cat((start,prediction),dim=2)
        return prediction.transpose(1,2)
