"""Trafformer predicts the configured horizon; gradient accumulation is external."""
import torch
from torch import nn
from libcity.model.traffic_speed_prediction.Trafformer import Informer_t_and_s


class ForecastInformer(Informer_t_and_s):
    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec):
        count,nodes,history,features = x_enc.shape
        length = x_dec.shape[2]
        encoded = self.enc_embedding(x_enc.reshape(count*nodes,history,features),x_mark_enc)
        encoded = (encoded.reshape(count,nodes,history,-1)+self.w).reshape(count,nodes*history,-1)
        encoded,_ = self.encoder(encoded,attn_mask=None)
        decoded = self.dec_embedding(x_dec.reshape(count*nodes,length,features),x_mark_dec)
        decoded = (decoded.reshape(count,nodes,length,self.d_model)+self.w).reshape(count,nodes*length,self.d_model)
        decoded = self.decoder(decoded,encoded,x_mask=None,cross_mask=None)
        decoded = decoded.reshape(count,nodes,length,-1)[:,:,history:]
        return decoded.permute(0,2,1,3)

    def _apply(self, function, recurse=True):
        result = super()._apply(function,recurse=recurse)
        for module in self.modules():
            if hasattr(module,'cache') and isinstance(module.cache,dict):
                module.cache.clear()
        return result


class Trafformer(nn.Module):
    def __init__(self, *,metadata,input_window,output_window,d_model=32,embed='timeF',freq='m',dropout=0.,
                 enc_attn='full',factor=50,n_heads=4,d_ff=256,activation='gelu',e_layers=2,distil=False,
                 mix=True,d_layers=1):
        super().__init__()
        self.output_dim,self.output_window = metadata.output_dim,output_window
        self.future_indices = tuple(metadata.input_layout.get('future_feature_indices', range(metadata.output_dim,metadata.feature_dim)))
        self.xformer = ForecastInformer(pred_len=output_window,output_attention=False,
            enc_in=self.output_dim,d_model=d_model,embed=embed,freq=freq,dropout=dropout,
            dec_in=self.output_dim,enc_attn=enc_attn,factor=factor,n_heads=n_heads,d_ff=d_ff,
            activation=activation,e_layers=e_layers,distil=distil,mix=mix,d_layers=d_layers,
            out_size=self.output_dim,num_nodes=metadata.num_nodes)

    def forward(self,batch):
        history = batch['X'].transpose(1,2)
        count,nodes,length,features = history.shape
        encoder = history[...,:self.output_dim]
        decoder = torch.cat((encoder,encoder.new_zeros(count,nodes,self.output_window,self.output_dim)),dim=2)
        mark_encoder = mark_decoder = None
        if self.future_indices:
            if 'future_features' not in batch:
                raise ValueError('Trafformer calendar channels require future_features')
            mark_encoder = history[...,list(self.future_indices)].reshape(count*nodes,length,-1)
            future = batch['future_features'].transpose(1,2).reshape(count*nodes,self.output_window,-1)
            mark_decoder = torch.cat((mark_encoder,future),dim=1)
        return self.xformer(encoder,mark_encoder,decoder,mark_decoder)
