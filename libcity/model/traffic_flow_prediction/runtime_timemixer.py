"""Multi-scale mixing with explicit temporal lengths and local normalization state."""
import torch
from torch import nn

from libcity.model.traffic_flow_prediction.TimeMixer import (
    TokenEmbedding,series_decomp,PastDecomposableMixing,
    MultiScaleSeasonMixing,MultiScaleTrendMixing)


class Normalize(nn.Module):
    def __init__(self,channels,enabled):
        super().__init__()
        self.enabled,self.eps = enabled,1e-5
        self.affine_weight = nn.Parameter(torch.ones(channels))
        self.affine_bias = nn.Parameter(torch.zeros(channels))

    def normalize(self,source):
        if not self.enabled:
            return source,None
        mean = source.mean(1,keepdim=True).detach()
        deviation = (source.var(1,keepdim=True,unbiased=False)+self.eps).sqrt().detach()
        return (source-mean)/deviation*self.affine_weight+self.affine_bias,(mean,deviation)

    def denormalize(self,prediction,statistics):
        if statistics is None:
            return prediction
        mean,deviation = statistics
        weight,bias = self.affine_weight,self.affine_bias
        if prediction.ndim==4:
            mean,deviation = mean.unsqueeze(-1),deviation.unsqueeze(-1)
            weight,bias = weight[None,None,:,None],bias[None,None,:,None]
        return (prediction-bias)/(weight+self.eps**2)*deviation+mean


class FrequencyDecomposition(nn.Module):
    def __init__(self,top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self,source):
        # FFT runs over time, independently for every batch and hidden channel.
        with torch.autocast(source.device.type,enabled=False):
            values = source.float() if source.dtype in (torch.float16,torch.bfloat16) else source
            spectrum = torch.fft.rfft(values,dim=1)
            count = min(self.top_k,spectrum.shape[1]-1)
            selected = torch.zeros_like(spectrum.real,dtype=torch.bool)
            if count:
                magnitude = spectrum.abs()
                magnitude[:,0] = -torch.inf
                indices = magnitude.topk(count,dim=1).indices
                selected.scatter_(1,indices,True)
            seasonal = torch.fft.irfft(torch.where(selected,spectrum,0.),n=source.shape[1],dim=1)
        seasonal = seasonal.to(source.dtype)
        return seasonal,source-seasonal


def temporal_projection(source_length,target_length):
    return nn.Sequential(nn.Linear(source_length,target_length),nn.GELU(),nn.Linear(target_length,target_length))


class SeasonMixing(MultiScaleSeasonMixing):
    def __init__(self,lengths):
        nn.Module.__init__(self)
        self.down_sampling_layers = nn.ModuleList(temporal_projection(a,b) for a,b in zip(lengths[:-1],lengths[1:]))

    def forward(self,values):
        return [values[0].transpose(1,2)] if len(values)==1 else super().forward(values)


class TrendMixing(MultiScaleTrendMixing):
    def __init__(self,lengths):
        nn.Module.__init__(self)
        self.up_sampling_layers = nn.ModuleList(temporal_projection(b,a) for a,b in reversed(list(zip(lengths[:-1],lengths[1:]))))

    def forward(self,values):
        return [values[0].transpose(1,2)] if len(values)==1 else super().forward(values)


class PastMixing(PastDecomposableMixing):
    def __init__(self,lengths,hidden,feedforward,independent,decomposition,kernel,top_k):
        nn.Module.__init__(self)
        self.channel_independence = independent
        self.decompsition = series_decomp(kernel) if decomposition=='moving_avg' else FrequencyDecomposition(top_k)
        cross = lambda: nn.Sequential(nn.Linear(hidden,feedforward),nn.GELU(),nn.Linear(feedforward,hidden))
        if independent:
            self.out_cross_layer = cross()
        else:
            self.cross_layer = cross()
        self.mixing_multi_scale_season = SeasonMixing(lengths)
        self.mixing_multi_scale_trend = TrendMixing(lengths)


class ValueEmbedding(nn.Module):
    def __init__(self,channels,hidden,dropout):
        super().__init__()
        self.value_embedding = TokenEmbedding(channels,hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self,source):
        return self.dropout(self.value_embedding(source))


class TimeMixer(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,down_sampling_layers=1,down_sampling_window=2,
                 down_sampling_method='avg',channel_independence=False,e_layers=5,d_model=128,d_ff=256,
                 use_norm=False,decomp_method='moving_avg',moving_avg=25,dropout=.1,top_k=5):
        super().__init__()
        if min(d_model,d_ff,down_sampling_window,moving_avg)<1 or min(down_sampling_layers,e_layers,top_k)<0:
            raise ValueError('TimeMixer requires positive widths/windows and nonnegative layer/frequency counts')
        if down_sampling_method not in ('avg','max','conv') or decomp_method not in ('moving_avg','dft_decomp') or not 0<=dropout<1:
            raise ValueError('TimeMixer requires avg/max/conv sampling, moving_avg/dft_decomp and valid dropout')
        if metadata.input_layout.get('kind')!='window':
            raise ValueError('TimeMixer requires contiguous input windows')
        self.nodes,self.horizon = metadata.num_nodes,output_window
        self.independent = channel_independence
        lengths = [input_window]
        for _ in range(down_sampling_layers):
            following = (lengths[-1]+down_sampling_window-1)//down_sampling_window if down_sampling_method=='conv' else lengths[-1]//down_sampling_window
            if following<1:
                raise ValueError('TimeMixer downsampling produces an empty scale; reduce down_sampling_layers/window')
            lengths.append(following)
        self.scale_lengths = tuple(lengths)
        if down_sampling_method=='conv':
            self.down_pool = nn.Conv1d(self.nodes,self.nodes,3,stride=down_sampling_window,padding=1,padding_mode='circular',bias=False)
        else:
            self.down_pool = (nn.AvgPool1d if down_sampling_method=='avg' else nn.MaxPool1d)(down_sampling_window)
        self.pdm_blocks = nn.ModuleList(PastMixing(lengths,d_model,d_ff,channel_independence,decomp_method,moving_avg,top_k)
                                       for _ in range(e_layers))
        self.preprocess = series_decomp(moving_avg)
        self.enc_embedding = ValueEmbedding(1 if channel_independence else self.nodes,d_model,dropout)
        self.predict_layers = nn.ModuleList(nn.Linear(length,output_window) for length in lengths)
        self.projection_layer = nn.Linear(d_model,1 if channel_independence else self.nodes)
        if not channel_independence:
            self.out_res_layers = nn.ModuleList(nn.Linear(length,length) for length in lengths)
            self.regression_layers = nn.ModuleList(nn.Linear(length,output_window) for length in lengths)
        self.normalize_layers = nn.ModuleList(Normalize(self.nodes,use_norm) for _ in lengths)
        self.pre_layer = nn.Linear(metadata.feature_dim,1)
        self.out_layer = nn.Identity() if channel_independence and metadata.output_dim==1 else nn.Linear(1,metadata.output_dim)

    def forward(self,batch):
        source = self.pre_layer(batch['X']).squeeze(-1)
        count = len(source)
        scales = [source]
        for _ in range(len(self.scale_lengths)-1):
            scales.append(self.down_pool(scales[-1].transpose(1,2)).transpose(1,2))
        normalized,statistics = [],[]
        for layer,values in zip(self.normalize_layers,scales):
            values,state = layer.normalize(values)
            statistics.append(state)
            normalized.append(values.transpose(1,2).reshape(count*self.nodes,values.shape[1],1) if self.independent else values)
        residuals = []
        if self.independent:
            seasonal = normalized
        else:
            pieces = [self.preprocess(values) for values in normalized]
            seasonal,residuals = [a for a,b in pieces],[b for a,b in pieces]
        encoded = [self.enc_embedding(values) for values in seasonal]
        for block in self.pdm_blocks:
            encoded = block(encoded)
        predictions = []
        for index,values in enumerate(encoded):
            prediction = self.projection_layer(self.predict_layers[index](values.transpose(1,2)).transpose(1,2))
            if self.independent:
                prediction = prediction.reshape(count,self.nodes,self.horizon).transpose(1,2)
            else:
                residual = self.out_res_layers[index](residuals[index].transpose(1,2))
                prediction = prediction+self.regression_layers[index](residual).transpose(1,2)
                prediction = self.out_layer(prediction.unsqueeze(-1))
            predictions.append(prediction)
        combined = self.normalize_layers[0].denormalize(torch.stack(predictions).sum(0),statistics[0])
        return self.out_layer(combined.unsqueeze(-1)) if self.independent else combined
