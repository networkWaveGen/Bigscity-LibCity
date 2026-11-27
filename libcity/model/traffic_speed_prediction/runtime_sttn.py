"""Spatial/temporal transformers with stateless graph normalization."""
import torch
from torch import nn

from libcity.model.traffic_speed_prediction.STTN import Transformer


class STTN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,embed_dim=64,num_layers=3,num_heads=2,
                 forward_expansion=4,dropout_rate=0.,position_size=0):
        super().__init__()
        if min(embed_dim,num_layers,num_heads,forward_expansion)<1 or embed_dim%num_heads or not 0<=dropout_rate<1:
            raise ValueError('STTN requires positive dimensions, embed_dim divisible by num_heads, and valid dropout')
        if metadata.input_layout.get('kind')!='window' or position_size<0:
            raise ValueError('STTN requires contiguous input and nonnegative position_size')
        interval = metadata.info.get('time_intervals',300)
        positions = position_size or max(input_window,86400//interval)
        if positions<input_window:
            raise ValueError('STTN position_size must cover input_window')
        self.conv1 = nn.Conv2d(metadata.feature_dim,embed_dim,1)
        self.transformer = Transformer(metadata.adjacency.copy(),embed_dim=embed_dim,num_layers=num_layers,
            num_heads=num_heads,TG_per_day=positions,forward_expansion=forward_expansion,dropout_rate=dropout_rate)
        self.conv2 = nn.Conv2d(input_window,output_window,1)
        self.conv3 = nn.Conv2d(embed_dim,metadata.output_dim,1)
        self.act_layer = nn.ReLU()

    def forward(self,batch):
        encoded = self.conv1(batch['X'].permute(0,3,2,1)).permute(0,2,3,1)
        encoded = self.transformer(encoded).permute(0,2,1,3)
        decoded = self.act_layer(self.conv2(encoded)).permute(0,3,2,1)
        return self.conv3(decoded).permute(0,3,2,1)
