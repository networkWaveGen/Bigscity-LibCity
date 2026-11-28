"""Dynamic local/global encoding with explicit sample, region and time axes."""
from collections import OrderedDict
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def spatial_encoding(coordinates,width):
    indices = torch.arange(width,dtype=torch.float64)
    rates = 10000**(-(2*torch.div(indices,2,rounding_mode='floor'))/width)
    values = coordinates.double()[...,None]*rates
    return torch.where(indices.remainder(2)==0,values[...,0,:].sin(),values[...,1,:].cos()).float()


def grid_patches(values,radius):
    """[B,T,H,W,C] -> [B,T,H*W,L,L,C], including boolean validity."""
    side = 2*radius+1
    image = F.pad(values.permute(0,1,4,2,3),(radius,radius,radius,radius))
    patches = image.unfold(3,side,1).unfold(4,side,1)
    return patches.permute(0,1,3,4,5,6,2).reshape(*values.shape[:2],-1,side,side,values.shape[-1])


def attention(query,key,value,blocked=None):
    logits = (query@key.transpose(-1,-2))/math.sqrt(key.shape[-1])
    logits = logits.float() if logits.dtype in (torch.float16,torch.bfloat16) else logits
    if blocked is not None:
        empty = blocked.all(-1,keepdim=True)
        logits = torch.where(empty,0.,logits.masked_fill(blocked,float('-inf')))
    weights = logits.softmax(-1)
    if blocked is not None:
        weights = weights.masked_fill(blocked,0.)
    return weights.to(value.dtype)@value


class MSA(nn.Module):
    def __init__(self,width,heads,self_att=True):
        super().__init__()
        self.heads,self.depth,self.self_att = heads,width//heads,self_att
        if self_att:
            self.wx = nn.Linear(width,3*width)
        else:
            self.wq,self.wkv = nn.Linear(width,width),nn.Linear(width,2*width)
        self.wo = nn.Linear(width,width)

    def split(self,value):
        return value.unflatten(-1,(self.heads,self.depth)).transpose(-3,-2)

    def forward(self,query,key=None,blocked=None):
        if self.self_att:
            query,key,value = self.wx(query).chunk(3,-1)
        else:
            query = self.wq(query)
            key,value = self.wkv(key).chunk(2,-1)
        result = attention(self.split(query),self.split(key),self.split(value),
            blocked.unsqueeze(-3) if blocked is not None else None)
        return self.wo(result.transpose(-3,-2).flatten(-2))


def feed_forward(width):
    return nn.Sequential(nn.Linear(width,4*width),nn.ReLU(),nn.Linear(4*width,width))


def external_encoder(features,width):
    return nn.Sequential(nn.Linear(features,4*width),nn.ReLU(),nn.Linear(4*width,width),nn.Sigmoid()) if features else None


class EncoderLayer(nn.Module):
    def __init__(self,width,heads,dropout):
        super().__init__()
        self.msa,self.ffn = MSA(width,heads),feed_forward(width)
        self.layernorm1,self.layernorm2 = nn.LayerNorm(width,eps=1e-6),nn.LayerNorm(width,eps=1e-6)
        self.dropout1,self.dropout2 = nn.Dropout(dropout),nn.Dropout(dropout)

    def forward(self,values,blocked):
        values = self.layernorm1(values+self.dropout1(self.msa(values,blocked=blocked)))
        return self.layernorm2(values+self.dropout2(self.ffn(values)))


class DecoderLayer(nn.Module):
    def __init__(self,width,heads,dropout,revert_q=False):
        super().__init__()
        self.revert_q = revert_q
        self.msa1,self.msa2 = MSA(width,heads),MSA(width,heads,False)
        self.ffn = feed_forward(width)
        self.layernorm1,self.layernorm2,self.layernorm3 = [nn.LayerNorm(width,eps=1e-6) for _ in range(3)]
        self.dropout1,self.dropout2,self.dropout3 = [nn.Dropout(dropout) for _ in range(3)]

    def forward(self,values,memory,self_mask,memory_mask):
        values = self.layernorm1(values+self.dropout1(self.msa1(values,blocked=self_mask)))
        query = values.transpose(-3,-2) if self.revert_q else values
        cross = self.msa2(query,memory,blocked=memory_mask)
        cross = cross.transpose(-3,-2) if self.revert_q else cross
        values = self.layernorm2(values+self.dropout2(cross))
        return self.layernorm3(values+self.dropout3(self.ffn(values)))


class Convs(nn.Module):
    def __init__(self,layers,width,steps,features,dropout):
        super().__init__()
        self.convs = nn.ModuleList([nn.ModuleList([nn.Conv2d(features if layer==0 else width,width,3,padding=1)
            for _ in range(steps)]) for layer in range(layers)])
        self.dropouts = nn.ModuleList([nn.ModuleList([nn.Dropout(dropout) for _ in range(steps)]) for _ in range(layers)])

    def forward(self,values):
        local = values.ndim==6
        result = []
        for time,frame in enumerate(values.unbind(1)):
            if local:
                frame = frame.flatten(0,1)
            frame = frame.permute(0,3,1,2)
            for convolutions,dropouts in zip(self.convs,self.dropouts):
                frame = dropouts[time](convolutions[time](frame).relu())
            frame = frame.permute(0,2,3,1)
            result.append(frame.reshape(values.shape[0],values.shape[2],-1,frame.shape[-1]) if local else frame.flatten(1,2))
        return torch.stack(result,1)


class DAE(nn.Module):
    def __init__(self,layers,width,heads,convolutions,steps,features,external,dropout):
        super().__init__()
        self.scale = math.sqrt(width)
        self.convs_d = Convs(convolutions,width,steps,features,dropout)
        self.convs_g = Convs(convolutions,width,steps,features,dropout)
        self.ex_encoder_d,self.ex_encoder_g = external_encoder(external,width),external_encoder(external,width)
        self.dropout_d,self.dropout_g = nn.Dropout(dropout),nn.Dropout(dropout)
        self.Enc_G = nn.ModuleList([EncoderLayer(width,heads,dropout) for _ in range(layers)])
        self.Enc_D = nn.ModuleList([DecoderLayer(width,heads,dropout) for _ in range(layers)])

    def forward(self,local,global_values,external,local_pos,global_pos,local_valid,global_valid):
        local = self.convs_d(local)*self.scale+local_pos
        global_values = self.convs_g(global_values)*self.scale+global_pos
        if self.ex_encoder_d is not None:
            local = local+self.ex_encoder_d(external).unsqueeze(-2)
            global_values = global_values+self.ex_encoder_g(external)
        local,global_values = self.dropout_d(local),self.dropout_g(global_values)
        global_mask = ~global_valid.unsqueeze(-2)
        for layer in self.Enc_G:
            global_values = layer(global_values,global_mask)
        # Keep B and N distinct. Global data broadcast over N within each B.
        local = local.permute(0,2,1,3,4)
        local_mask = ~local_valid.permute(0,2,1,3).unsqueeze(-2)
        for layer in self.Enc_D:
            local = layer(local,global_values.unsqueeze(1),local_mask,global_mask.unsqueeze(1))
        return local


class SAD(nn.Module):
    def __init__(self,layers,width,heads,convolutions,steps,horizon,features,external,dropout):
        super().__init__()
        self.scale = math.sqrt(width)
        self.ex_encoder = external_encoder(external,width)
        self.dropout = nn.Dropout(dropout)
        modules = [('linear',nn.Linear(features,width)),('activation_relu',nn.ReLU())]
        for index in range(convolutions-1):
            modules += [(f'linear{index}',nn.Linear(width,width)),(f'activation_relu{index}',nn.ReLU())]
        self.li_conv = nn.Sequential(OrderedDict(modules))
        self.dec_s = nn.ModuleList([DecoderLayer(width,heads,dropout) for _ in range(layers)])
        self.linear = nn.Linear(steps,horizon)
        self.dec_t = nn.ModuleList([DecoderLayer(width,heads,dropout,True) for _ in range(layers)])
        self.register_buffer('pos_enc',spatial_encoding(torch.zeros(2),width))
        self.register_buffer('causal_mask',torch.ones(horizon,horizon,dtype=torch.bool).triu(1))

    def forward(self,seeds,external,memory,seed_valid):
        values = self.li_conv(seeds)*self.scale+self.pos_enc
        if self.ex_encoder is not None:
            values = values+self.ex_encoder(external)
        values = self.dropout(values).permute(0,2,1,3)
        horizon = values.shape[-2]
        spatial = values.unsqueeze(-2).expand(-1,-1,-1,horizon,-1)
        projected = self.linear(memory.movedim(2,-1)).movedim(-1,2)
        blocked = self.causal_mask|(~seed_valid.permute(0,2,1)[:,:,None,None,:])
        for layer in self.dec_s:
            spatial = layer(spatial,projected,blocked,None)
        temporal = values.unsqueeze(2)
        for layer in self.dec_t:
            temporal = layer(temporal,spatial.transpose(-3,-2),blocked,None)
        return temporal.squeeze(2)


class DsanUse(nn.Module):
    def __init__(self,layers,width,heads,convolutions,steps,horizon,features,external,dropout):
        super().__init__()
        self.dae = DAE(layers,width,heads,convolutions,steps,features,external,dropout)
        self.sad = SAD(layers,width,heads,convolutions,steps,horizon,features,external,dropout)
        self.final_layer = nn.Linear(width,features)

    def forward(self,local,global_values,external,local_pos,global_pos,local_valid,global_valid,seeds,seed_external,seed_valid):
        encoded = self.dae(local,global_values,external,local_pos,global_pos,local_valid,global_valid)
        decoded = self.sad(seeds,seed_external,encoded,seed_valid)
        return self.final_layer(decoded).tanh().transpose(1,2)


def decoder_seed(values,horizon):
    if horizon<=values.shape[1]:
        return values[:,-horizon:]
    return torch.cat((values,values[:,-1:].expand(-1,horizon-values.shape[1],*values.shape[2:])),1)


class DSAN(nn.Module):
    def __init__(self,*,metadata,input_window,output_window,L=3,d=64,n_h=8,B=3,l_d=2,r_d=.1):
        super().__init__()
        for name,value in (('input_window',input_window),('output_window',output_window),('L',L),('d',d),('n_h',n_h),('B',B)):
            if type(value) is not int or value<1:
                raise ValueError(f'DSAN {name} must be a positive integer')
        if d%n_h or type(l_d) is not int or l_d<0 or type(r_d) not in (int,float) or not 0<=r_d<1:
            raise ValueError('DSAN requires d divisible by n_h, nonnegative l_d and dropout in [0,1)')
        layout = metadata.input_layout
        if layout.get('kind')!='grid_window' or layout.get('history_mask')!='finite_nonzero_raw_observations':
            raise ValueError('DSAN requires grid_window data with a raw history mask')
        shape = layout.get('grid_shape',())
        if len(shape)!=2 or any(type(value) is not int or value<1 for value in shape):
            raise ValueError('DSAN requires a complete rectangular grid')
        self.height,self.width = shape
        self.nodes,self.features,self.outputs = metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        order,positions = (np.array(layout.get(name,()),copy=True) for name in ('grid_order','node_positions'))
        if (self.nodes!=self.height*self.width or not 1<=self.outputs<=self.features or
                order.dtype.kind not in 'iu' or positions.dtype.kind not in 'iu' or
                order.shape!=(self.nodes,) or positions.shape!=(self.nodes,) or
                not np.array_equal(np.sort(order),np.arange(self.nodes)) or
                not np.array_equal(np.sort(positions),np.arange(self.nodes)) or
                not np.array_equal(order[positions],np.arange(self.nodes))):
            raise ValueError('DSAN grid order and feature dimensions must match metadata')
        self.external = layout.get('historical_external_dim')
        if type(self.external) is not int or self.external<0:
            raise ValueError('DSAN requires historical_external_dim metadata')
        self.register_buffer('grid_order',torch.tensor(order,dtype=torch.long))
        self.register_buffer('node_positions',torch.tensor(positions,dtype=torch.long))
        self.input_window,self.output_window,self.radius = input_window,output_window,l_d
        rows,columns = torch.meshgrid(torch.arange(self.height)-self.height//2,torch.arange(self.width)-self.width//2,indexing='ij')
        self.register_buffer('global_positions',spatial_encoding(torch.stack((rows,columns),-1).flatten(0,1),d)[None,None])
        rows,columns = torch.meshgrid(torch.arange(-l_d,l_d+1),torch.arange(-l_d,l_d+1),indexing='ij')
        self.register_buffer('local_positions',spatial_encoding(torch.stack((rows,columns),-1).flatten(0,1),d)[None,None,None])
        self.dsan = DsanUse(L,d,n_h,B,input_window,output_window,self.outputs,self.features-self.outputs+self.external,r_d)

    def forward(self,inputs):
        values,mask = inputs['X'],inputs.get('history_mask')
        if values.ndim!=4 or values.shape[1:]!=(self.input_window,self.nodes,self.features):
            raise ValueError('DSAN X must match [B,input_window,N,feature_dim]')
        if mask is None or mask.shape!=values.shape or mask.dtype!=torch.bool:
            raise ValueError('DSAN history_mask must be boolean and match X')
        values = values.index_select(2,self.grid_order)
        valid = mask.index_select(2,self.grid_order)[...,:self.outputs].any(-1)
        external = values[...,self.outputs:]
        if self.external:
            observed = inputs.get('history_features')
            if observed is None or observed.shape!=(*values.shape[:2],self.external):
                raise ValueError('DSAN history_features must match global historical covariates')
            external = torch.cat((external,observed[:,:,None].expand(-1,-1,self.nodes,-1)),-1)
        flow = values[...,:self.outputs]
        grid = flow.reshape(*flow.shape[:2],self.height,self.width,self.outputs)
        local = grid_patches(grid,self.radius)
        local_valid = grid_patches(valid.reshape(*valid.shape[:2],self.height,self.width,1),self.radius).flatten(3).bool()
        prediction = self.dsan(local,grid,external,self.local_positions,self.global_positions,local_valid,valid,
            decoder_seed(flow,self.output_window),decoder_seed(external,self.output_window),decoder_seed(valid,self.output_window))
        return prediction.index_select(2,self.node_positions)
