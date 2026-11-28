"""Grid forecasts with explicit time/feature/spatial axes and no target inputs."""
from collections import OrderedDict
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.traffic_flow_prediction.ACFM import ConvLSTM, ExtNN, ConcatConv, ResUnit
from libcity.model.runtime_norm import SingletonBatchNorm2d as GridBatchNorm


def safe_batchnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module,name,GridBatchNorm(child.num_features,eps=child.eps,momentum=child.momentum))
        else:
            safe_batchnorm(child)
    return module


class GridModel(nn.Module):
    def __init__(self, metadata, output_window):
        super().__init__()
        layout = metadata.input_layout
        if layout.get('kind') != 'grid_cpt':
            raise ValueError('Grid forecasts require the grid_cpt data adapter')
        self.height,self.width = layout['grid_shape']
        self.features,self.outputs,self.horizon = metadata.feature_dim,metadata.output_dim,output_window
        self.counts = tuple(layout['sampling'].get(name,0) for name in ('closeness','period','trend'))
        self.length = sum(self.counts)
        self.history_ext = layout['historical_external_dim']
        self.future_ext = layout['future_external_dim']
        self.register_buffer('grid_order',torch.tensor(np.array(layout['grid_order'],copy=True),dtype=torch.long))
        self.register_buffer('node_positions',torch.tensor(np.array(layout['node_positions'],copy=True),dtype=torch.long))

    def source(self,batch):
        source = batch['X']
        if source.ndim != 4 or source.shape[1:] != (self.length,self.height*self.width,self.features):
            raise ValueError('Grid input must have declared [batch,CPT time,node,feature] axes')
        return source[:,:,self.grid_order].reshape(len(source),self.length,self.height,self.width,self.features).permute(0,1,4,2,3)

    def output(self,prediction):
        prediction = prediction.reshape(-1,self.horizon,self.outputs,self.height,self.width)
        prediction = prediction.permute(0,1,3,4,2).reshape(-1,self.horizon,self.height*self.width,self.outputs)
        return prediction[:,:,self.node_positions]


class BnReluConv(nn.Module):
    def __init__(self,filters,bn):
        super().__init__()
        self.bn1 = GridBatchNorm(filters) if bn else nn.Identity()
        self.conv1 = nn.Conv2d(filters,filters,3,padding=1)

    def forward(self,source):
        return self.conv1(F.relu(self.bn1(source)))


class ResidualUnit(nn.Module):
    def __init__(self,filters,bn):
        super().__init__()
        self.bn_relu_conv1,self.bn_relu_conv2 = BnReluConv(filters,bn),BnReluConv(filters,bn)

    def forward(self,source):
        return source+self.bn_relu_conv2(self.bn_relu_conv1(source))


class ResidualStack(nn.Module):
    def __init__(self,filters,repetitions,bn):
        super().__init__()
        self.stacked_resunits = nn.Sequential(*(ResidualUnit(filters,bn) for _ in range(repetitions)))

    def forward(self,source):
        return self.stacked_resunits(source)


class Fusion(nn.Module):
    def __init__(self,channels,height,width):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,channels,height,width))

    def forward(self,source):
        return source*self.weights


class STResNet(GridModel):
    def __init__(self,*,metadata,input_window,output_window,nb_residual_unit=12,batch_norm=False,filters=64):
        super().__init__(metadata,output_window)
        if any(type(value) is not int or value<1 for value in (nb_residual_unit,filters)) or type(batch_norm) is not bool:
            raise ValueError('STResNet requires positive residual/filter counts and boolean batch_norm')
        self.segments = []
        start = 0
        for name,count in zip(('c_way','p_way','t_way'),self.counts):
            if count:
                setattr(self,name,nn.Sequential(OrderedDict([
                    ('conv1',nn.Conv2d(count*self.features,filters,3,padding=1)),
                    ('ResUnits',ResidualStack(filters,nb_residual_unit,batch_norm)),
                    ('relu',nn.ReLU()),('conv2',nn.Conv2d(filters,self.horizon*self.outputs,3,padding=1)),
                    ('FusionLayer',Fusion(self.horizon*self.outputs,self.height,self.width))])))
                self.segments.append((name,start,start+count))
            start += count
        if self.future_ext:
            self.external_ops = nn.Sequential(OrderedDict([
                ('embd',nn.Linear(self.future_ext,10)),('relu1',nn.ReLU()),
                ('fc',nn.Linear(10,self.outputs*self.height*self.width)),('relu2',nn.ReLU())]))

    def forward(self,batch):
        source = self.source(batch)
        predictions = [getattr(self,name)(source[:,start:end].flatten(1,2)) for name,start,end in self.segments]
        prediction = torch.stack(predictions).sum(0)
        if self.future_ext:
            external = self.external_ops(batch['future_features']).reshape(-1,self.horizon*self.outputs,self.height,self.width)
            prediction = prediction+external
        return self.output(torch.tanh(prediction))


class ResidualFeatures(nn.Module):
    def __init__(self,features,hidden,filters,repetitions,bn,mode,counts):
        super().__init__()
        steps = sum(counts)
        self.resblocks = nn.ModuleList()
        def block(inputs,outputs):
            return nn.Sequential(nn.Conv2d(inputs,filters,3,padding=1),
                *(safe_batchnorm(ResUnit(filters,bn)) for _ in range(repetitions)),
                nn.SELU(inplace=True),nn.Conv2d(filters,outputs,3,padding=1))
        if mode in ('split','split-chans'):
            pieces = steps if mode=='split' else steps*features
            if hidden*steps % pieces:
                raise ValueError('ACFM split-chans requires lstm_channels divisible by observation feature count')
            self.inlist = [steps*features//pieces]*pieces
            shared = block(self.inlist[0],hidden*steps//pieces)
            self.resblocks.extend(shared for _ in range(pieces))
        elif mode=='concat':
            self.inlist = [steps*features]
            self.resblocks.append(block(steps*features,steps*hidden))
        elif mode in ('cpt','cpt-sameoutput'):
            if mode=='cpt-sameoutput' and any(count*hidden != 2 for count in counts if count):
                raise ValueError('ACFM cpt-sameoutput requires each active CPT count * lstm_channels == 2')
            self.inlist = [count*features for count in counts if count]
            self.resblocks.extend(block(count*features,count*hidden) for count in counts if count)
        else:
            raise ValueError('ACFM res_split_mode must be split, split-chans, concat, cpt or compatible cpt-sameoutput')

    def forward(self,source):
        return torch.cat([block(value) for block,value in zip(self.resblocks,source.split(self.inlist,1))],1)


class ACFM(GridModel):
    def __init__(self,*,metadata,input_window,output_window,res_repetation=12,res_nbfilter=16,res_bn=True,
                 res_split_mode='split',first_extnn_inter_channels=40,first_extnn_dropout=.5,lstm_channels=16,lstm_dropout=0.):
        super().__init__(metadata,output_window)
        values = (res_repetation,res_nbfilter,first_extnn_inter_channels,lstm_channels)
        if any(type(value) is not int or value<1 for value in values) or type(res_bn) is not bool:
            raise ValueError('ACFM requires positive dimensions and boolean res_bn')
        if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<1
               for value in (first_extnn_dropout,lstm_dropout)):
            raise ValueError('ACFM dropout rates must be finite and in [0,1)')
        self.channels = lstm_channels
        self.local,self.global_ = self.counts[0],sum(self.counts[1:])
        self.resnn = ResidualFeatures(self.features,lstm_channels,res_nbfilter,res_repetation,res_bn,res_split_mode,self.counts)
        self.conv_lstm = ConvLSTM(lstm_channels,self.height,self.width,lstm_channels,all_hidden=True,
            mode='cpt',cpt=[self.local,self.global_,0],dropout_rate=lstm_dropout)
        for suffix,count in (('c',self.local),('t',self.global_)):
            if count:
                setattr(self,'concat_conv_'+suffix,ConcatConv(lstm_channels*count,lstm_channels*count,
                    lstm_channels*count,lstm_channels,relu_conv=True,seq_len=count))
                setattr(self,'conv_lstm_'+suffix,ConvLSTM(lstm_channels,self.height,self.width,lstm_channels,
                    dropout_rate=lstm_dropout,last_conv=True,conv_channels=self.horizon*self.outputs))
        if self.history_ext:
            self.extnn = ExtNN(self.history_ext,self.height,self.width,lstm_channels,
                inter_features=first_extnn_inter_channels,dropout_rate=first_extnn_dropout)
        if self.future_ext:
            self.time_aware_extnn = ExtNN(self.future_ext,1,1,1,inter_features=32,map=False,relu=False,mode='last')

    def forward(self,batch):
        features = self.resnn(self.source(batch).flatten(1,2))
        if self.history_ext:
            features = features+self.extnn(batch['history_features'])
        states = self.conv_lstm(features)
        predictions = {}
        start = 0
        for suffix,count in (('c',self.local),('t',self.global_)):
            if count:
                end = start+count*self.channels
                original,hidden = features[:,start:end],states[:,start:end]
                attention = getattr(self,'concat_conv_'+suffix)(original,hidden)
                predictions[suffix] = getattr(self,'conv_lstm_'+suffix)(original*(1+attention))
                start = end
        if self.future_ext:
            gate = torch.sigmoid(self.time_aware_extnn(batch['future_features'].flatten(0,1)))
            gate = gate.reshape(len(features),self.horizon,1,1,1)
            template = next(iter(predictions.values())).reshape(-1,self.horizon,self.outputs,self.height,self.width)
            prediction = torch.zeros_like(template)
            for name,value in predictions.items():
                prediction = prediction+(gate if name=='c' else 1-gate)*value.reshape_as(template)
        else:
            prediction = torch.stack(list(predictions.values())).mean(0)
        return self.output(torch.tanh(prediction))
