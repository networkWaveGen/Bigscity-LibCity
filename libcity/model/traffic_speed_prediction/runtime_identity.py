"""Identity and attention forecasts with explicit calendar-channel metadata."""
import torch
from torch import nn

from libcity.model.traffic_speed_prediction.STID import MultiLayerPerceptron
from libcity.model.traffic_speed_prediction.STAEformer import SelfAttentionLayer


def calendar_metadata(metadata, time, week):
    calendar = metadata.input_layout.get('calendar', {})
    tid, dow = calendar.get('time_in_day'), tuple(calendar.get('day_of_week', ()))
    if (time and tid is None) or (week and len(dow) != 7):
        raise ValueError('Calendar embeddings require the corresponding data.features time/day channels')
    return tid, dow


def steps_in_day(metadata, requested=0):
    interval = metadata.info.get('time_intervals', 300)
    if requested:
        return requested
    if interval <= 0 or 86400 % interval:
        raise ValueError('Calendar embedding requires time_intervals to divide one day or explicit steps_per_day')
    return int(86400 // interval)


class STID(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, num_block=3, time_series_emb_dim=32,
                 spatial_emb_dim=32, temp_dim_tid=32, temp_dim_diw=32,
                 if_spatial=True, if_TiD=True, if_DiW=True):
        super().__init__()
        if (time_series_emb_dim < 1 or min(spatial_emb_dim, temp_dim_tid, temp_dim_diw, num_block) < 0
                or if_spatial and spatial_emb_dim < 1 or if_TiD and temp_dim_tid < 1 or if_DiW and temp_dim_diw < 1):
            raise ValueError('STID requires positive enabled embeddings and nonnegative disabled dimensions/num_block')
        self.output_dim, self.output_window = metadata.output_dim, output_window
        self.if_spatial, self.if_time_in_day, self.if_day_in_week = if_spatial, if_TiD, if_DiW
        self.tid_index, self.dow_indices = calendar_metadata(metadata, if_TiD, if_DiW)
        self.time_of_day_size = steps_in_day(metadata)
        if if_spatial:
            self.node_emb = nn.Parameter(torch.empty(metadata.num_nodes, spatial_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if if_TiD:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if if_DiW:
            self.day_in_week_emb = nn.Parameter(torch.empty(7, temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        self.time_series_emb_layer = nn.Conv2d(metadata.output_dim*input_window, time_series_emb_dim, 1)
        width = time_series_emb_dim + spatial_emb_dim*if_spatial + temp_dim_tid*if_TiD + temp_dim_diw*if_DiW
        self.encoder = nn.Sequential(*(MultiLayerPerceptron(width, width) for _ in range(num_block)))
        self.regression_layer = nn.Conv2d(width, output_window*metadata.output_dim, 1)

    def forward(self, batch):
        source = batch['X']
        count, _, nodes, _ = source.shape
        series = source[..., :self.output_dim].transpose(1, 2).reshape(count, nodes, -1).transpose(1, 2).unsqueeze(-1)
        features = [self.time_series_emb_layer(series)]
        if self.if_spatial:
            features.append(self.node_emb.T.unsqueeze(0).expand(count, -1, -1).unsqueeze(-1))
        if self.if_time_in_day:
            index = (source[:, -1, :, self.tid_index]*self.time_of_day_size).long().clamp(0, self.time_of_day_size-1)
            features.append(self.time_in_day_emb[index].transpose(1, 2).unsqueeze(-1))
        if self.if_day_in_week:
            index = source[:, -1, :, list(self.dow_indices)].argmax(dim=-1)
            features.append(self.day_in_week_emb[index].transpose(1, 2).unsqueeze(-1))
        prediction = self.regression_layer(self.encoder(torch.cat(features, dim=1)))
        return prediction[..., 0].reshape(count, self.output_window, self.output_dim, nodes).transpose(2, 3)


class STAEformer(nn.Module):
    def __init__(self, *, metadata, input_window, output_window, input_dim=0, steps_per_day=0,
                 input_embedding_dim=24, tod_embedding_dim=24, dow_embedding_dim=24,
                 spatial_embedding_dim=0, adaptive_embedding_dim=80, feed_forward_dim=256,
                 num_heads=4, num_layers=3, dropout=.1, use_mixed_proj=True):
        super().__init__()
        if min(input_embedding_dim, feed_forward_dim, num_heads, num_layers) < 1 or min(
                tod_embedding_dim, dow_embedding_dim, spatial_embedding_dim, adaptive_embedding_dim, input_dim, steps_per_day) < 0:
            raise ValueError('Invalid STAEformer dimensions')
        self.tid_index, self.dow_indices = calendar_metadata(metadata, tod_embedding_dim > 0, dow_embedding_dim > 0)
        excluded = set(self.dow_indices) | ({self.tid_index} if self.tid_index is not None else set())
        self.numeric_indices = tuple(index for index in range(metadata.feature_dim) if index not in excluded)
        self.input_dim = input_dim or len(self.numeric_indices)+int(self.tid_index is not None)+int(bool(self.dow_indices))
        packed_width = len(self.numeric_indices)+int(self.tid_index is not None)+int(bool(self.dow_indices))
        if self.input_dim > packed_width:
            raise ValueError('STAEformer input_dim exceeds packed numeric/calendar feature count')
        self.in_steps, self.out_steps, self.num_nodes = input_window, output_window, metadata.num_nodes
        self.output_dim, self.use_mixed_proj = metadata.output_dim, use_mixed_proj
        self.steps_per_day = steps_in_day(metadata, steps_per_day)
        self.model_dim = input_embedding_dim+tod_embedding_dim+dow_embedding_dim+spatial_embedding_dim+adaptive_embedding_dim
        if self.model_dim % num_heads:
            raise ValueError('STAEformer total embedding dimension must be divisible by num_heads')
        self.input_proj = nn.Linear(self.input_dim, input_embedding_dim)
        if tod_embedding_dim:
            self.tod_embedding = nn.Embedding(self.steps_per_day, tod_embedding_dim)
        if dow_embedding_dim:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim:
            self.node_emb = nn.Parameter(torch.empty(metadata.num_nodes, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim:
            self.adaptive_embedding = nn.Parameter(torch.empty(input_window, metadata.num_nodes, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        if use_mixed_proj:
            self.output_proj = nn.Linear(input_window*self.model_dim, output_window*metadata.output_dim)
        else:
            self.temporal_proj = nn.Linear(input_window, output_window)
            self.output_proj = nn.Linear(self.model_dim, metadata.output_dim)
        self.attn_layers_t = nn.ModuleList(SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers))
        self.attn_layers_s = nn.ModuleList(SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers))

    def packed_inputs(self, source):
        values = [source[..., list(self.numeric_indices)]]
        if self.tid_index is not None:
            values.append(source[..., self.tid_index:self.tid_index+1])
        if self.dow_indices:
            values.append(source[..., list(self.dow_indices)].argmax(dim=-1, keepdim=True).to(source.dtype))
        return torch.cat(values, dim=-1)

    def forward(self, batch):
        source = batch['X']
        count = source.shape[0]
        features = [self.input_proj(self.packed_inputs(source)[..., :self.input_dim])]
        if hasattr(self, 'tod_embedding'):
            index = (source[..., self.tid_index]*self.steps_per_day).long().clamp(0, self.steps_per_day-1)
            features.append(self.tod_embedding(index))
        if hasattr(self, 'dow_embedding'):
            features.append(self.dow_embedding(source[..., list(self.dow_indices)].argmax(dim=-1)))
        if hasattr(self, 'node_emb'):
            features.append(self.node_emb.expand(count, self.in_steps, -1, -1))
        if hasattr(self, 'adaptive_embedding'):
            features.append(self.adaptive_embedding.expand(count, -1, -1, -1))
        hidden = torch.cat(features, dim=-1)
        for attention in self.attn_layers_t:
            hidden = attention(hidden, dim=1)
        for attention in self.attn_layers_s:
            hidden = attention(hidden, dim=2)
        if self.use_mixed_proj:
            output = self.output_proj(hidden.transpose(1, 2).reshape(count, self.num_nodes, -1))
            return output.reshape(count, self.num_nodes, self.out_steps, self.output_dim).transpose(1, 2)
        return self.output_proj(self.temporal_proj(hidden.transpose(1, 3)).transpose(1, 3))
