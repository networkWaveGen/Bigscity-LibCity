"""Decoupled diffusion/inherent forecasts with shared, unexpanded graph orders."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libcity.model.runtime_norm import SingletonBatchNorm1d
from libcity.model.traffic_speed_prediction.runtime_identity import calendar_metadata,steps_in_day


def transition_supports(adjacency,nodes):
    graph=np.array(adjacency,dtype=np.float64,copy=True)
    if graph.shape!=(nodes,nodes) or not np.isfinite(graph).all() or np.any(graph<0):
        raise ValueError('D2STGNN requires finite nonnegative adjacency aligned to nodes')
    result=[]
    for value in (graph,graph.T):
        degree=value.sum(1,keepdims=True)
        result.append(np.divide(value,degree,out=np.zeros_like(value),where=degree>0).T)
    return torch.tensor(np.stack(result),dtype=torch.float32)


def graph_orders(graph,order,mask):
    result=[];power=graph
    for index in range(order):
        if index: power=power@graph
        result.append(power*mask)
    return result


class ResidualDecomp(nn.Module):
    def __init__(self,width):
        super().__init__();self.ln=nn.LayerNorm(width)
    def forward(self,source,backcast): return self.ln(source-backcast.relu())


class EstimationGate(nn.Module):
    def __init__(self,inputs):
        super().__init__();self.fully_connected_layer_1=nn.Linear(inputs,64);self.fully_connected_layer_2=nn.Linear(64,1)
    def forward(self,source,embeddings):
        features=embeddings[:,-source.shape[1]:]
        return source*self.fully_connected_layer_2(self.fully_connected_layer_1(features).relu()).sigmoid()


class STLocalizedConv(nn.Module):
    def __init__(self,hidden,kernel,supports,dropout):
        super().__init__();self.kernel=kernel
        self.fc_list_updt=nn.Linear(kernel*hidden,kernel*hidden,bias=False)
        self.gcn_updt=nn.Linear(hidden*(supports+1),hidden);self.dropout=nn.Dropout(dropout)
    def forward(self,source,supports):
        window=source.unfold(1,self.kernel,1).transpose(-1,-2)
        values=self.fc_list_updt(window.flatten(-2)).relu().reshape_as(window)
        # Repeated [A,...,A] times [X_t1;...;X_tk] equals A @ sum_t X_t.
        # Keep graph orders as N x N rather than allocating N x (k_t*N).
        temporal=values.sum(-2);parts=[values.mean(-2)]
        with torch.autocast(source.device.type,enabled=False):
            temporal=temporal.float() if temporal.dtype in (torch.float16,torch.bfloat16) else temporal
            for graph in supports:
                parts.append((graph[:,None] if graph.ndim==3 else graph)@temporal)
        return self.dropout(self.gcn_updt(torch.cat(parts,-1)))


class DifForecast(nn.Module):
    def __init__(self,hidden,forecast,chunks,kernel):
        super().__init__();self.chunks,self.kernel=chunks,kernel;self.forecast_fc=nn.Linear(hidden,forecast)
    def forward(self,history,hidden,convolution,supports):
        predicted=[hidden[:,-1:]]
        for _ in range(1,self.chunks):
            recent=predicted[-self.kernel:]
            if len(recent)<self.kernel: recent=[history[:,-(self.kernel-len(recent)):]]+recent
            predicted.append(convolution(torch.cat(recent,1),supports))
        return self.forecast_fc(torch.cat(predicted,1))


class DifBlock(nn.Module):
    def __init__(self,hidden,forecast,chunks,kernel,supports,dropout):
        super().__init__()
        self.localized_st_conv=STLocalizedConv(hidden,kernel,supports,dropout)
        self.forecast_branch=DifForecast(hidden,forecast,chunks,kernel)
        self.backcast_branch=nn.Linear(hidden,hidden);self.residual_decompose=ResidualDecomp(hidden)
    def forward(self,history,gated,supports):
        hidden=self.localized_st_conv(gated,supports)
        prediction=self.forecast_branch(gated,hidden,self.localized_st_conv,supports)
        return self.residual_decompose(history[:,-hidden.shape[1]:],self.backcast_branch(hidden)),prediction


class PositionalEncoding(nn.Module):
    def __init__(self,hidden,dropout,length):
        super().__init__();self.dropout=nn.Dropout(dropout)
        positions=torch.arange(length)[:,None];scale=torch.exp(torch.arange(0,hidden,2)*(-math.log(10000.)/hidden))
        values=torch.zeros(length,1,hidden)
        values[:,0,0::2]=(positions*scale).sin();values[:,0,1::2]=(positions*scale[:hidden//2]).cos()
        self.register_buffer('pe',values)
    def forward(self,source): return self.dropout(source+self.pe[:len(source)])


class RNNLayer(nn.Module):
    def __init__(self,hidden,dropout):
        super().__init__();self.gru_cell=nn.GRUCell(hidden,hidden);self.dropout=nn.Dropout(dropout)
    def forward(self,source):
        batch,steps,nodes,hidden=source.shape
        source=source.transpose(1,2).reshape(batch*nodes,steps,hidden)
        state=torch.zeros_like(source[:,0]);result=[]
        for value in source.unbind(1):
            state=self.gru_cell(value,state);result.append(state)
        return self.dropout(torch.stack(result))


class TransformerLayer(nn.Module):
    def __init__(self,hidden,heads,dropout):
        super().__init__();self.multi_head_self_attention=nn.MultiheadAttention(hidden,heads,dropout=dropout)
        self.dropout=nn.Dropout(dropout)
    def forward(self,query,key,value):
        return self.dropout(self.multi_head_self_attention(query,key,value,need_weights=False)[0])


class InhForecast(nn.Module):
    def __init__(self,hidden,forecast,chunks):
        super().__init__();self.chunks=chunks;self.forecast_fc=nn.Linear(hidden,forecast)
    def forward(self,source,recurrent,attention,transformer,rnn,position):
        predicted=[attention[-1:]]
        for _ in range(1,self.chunks):
            following=rnn.gru_cell(predicted[-1][0],recurrent[-1])[None]
            recurrent=position(torch.cat((recurrent,following)))
            # Preserve the author's recurrent positional-encoding order.
            predicted.append(transformer(following,recurrent,recurrent))
        values=torch.cat(predicted).reshape(self.chunks,len(source),source.shape[2],source.shape[3]).transpose(0,1)
        return self.forecast_fc(values)


class InhBlock(nn.Module):
    def __init__(self,hidden,forecast,chunks,heads,dropout,length,last):
        super().__init__();self.last=last
        self.pos_encoder=PositionalEncoding(hidden,dropout,length)
        self.rnn_layer=RNNLayer(hidden,dropout);self.transformer_layer=TransformerLayer(hidden,heads,dropout)
        self.forecast_block=InhForecast(hidden,forecast,chunks)
        if not last:
            self.backcast_fc=nn.Linear(hidden,hidden);self.residual_decompose=ResidualDecomp(hidden)
    def forward(self,source):
        recurrent=self.pos_encoder(self.rnn_layer(source))
        attention=self.transformer_layer(recurrent,recurrent,recurrent)
        prediction=self.forecast_block(source,recurrent,attention,self.transformer_layer,self.rnn_layer,self.pos_encoder)
        if self.last: return None,prediction
        hidden=attention.reshape(source.shape[1],len(source),source.shape[2],source.shape[3]).transpose(0,1)
        return self.residual_decompose(source,self.backcast_fc(hidden)),prediction


class DecoupleLayer(nn.Module):
    def __init__(self,hidden,forecast,chunks,kernel,supports,gate_inputs,heads,dropout,length,last):
        super().__init__();self.estimation_gate=EstimationGate(gate_inputs)
        self.dif_layer=DifBlock(hidden,forecast,chunks,kernel,supports,dropout)
        self.inh_layer=InhBlock(hidden,forecast,chunks,heads,dropout,length,last)
    def forward(self,source,supports,embeddings):
        inherent,diffusion=self.dif_layer(source,self.estimation_gate(source,embeddings),supports)
        remaining,forecast=self.inh_layer(inherent)
        return remaining,diffusion,forecast


class DistanceFunction(nn.Module):
    def __init__(self,history,hidden,node_dim,time_dim,dropout):
        super().__init__();self.hidden=hidden
        self.fc_ts_emb1=nn.Linear(history,2*hidden);self.fc_ts_emb2=nn.Linear(2*hidden,hidden)
        self.bn=SingletonBatchNorm1d(2*hidden);self.dropout=nn.Dropout(dropout)
        self.WQ=nn.Linear(hidden+node_dim+time_dim,hidden,bias=False)
        self.WK=nn.Linear(hidden+node_dim+time_dim,hidden,bias=False)
    def forward(self,source,embedding_d,embedding_u,times):
        batch,history,nodes=source.shape
        values=source.transpose(1,2).reshape(batch*nodes,history)
        hidden=self.fc_ts_emb2(self.dropout(self.bn(self.fc_ts_emb1(values).relu()))).reshape(batch,nodes,self.hidden)
        result=[]
        for embedding in (embedding_d,embedding_u):
            features=torch.cat([hidden,*[value[:,-1] for value in times],embedding[None].expand(batch,-1,-1)],-1)
            query,key=self.WQ(features),self.WK(features)
            with torch.autocast(source.device.type,enabled=False):
                if query.dtype in (torch.float16,torch.bfloat16): query,key=query.float(),key.float()
                result.append((query@key.transpose(-1,-2)/math.sqrt(self.hidden)).softmax(-1))
        return result


class DynamicGraphConstructor(nn.Module):
    def __init__(self,history,hidden,node_dim,time_dim,dropout,order):
        super().__init__();self.order=order
        self.distance_function=DistanceFunction(history,hidden,node_dim,time_dim,dropout)
    def forward(self,source,embedding_d,embedding_u,times,road,mask):
        result=[]
        for index,value in enumerate(self.distance_function(source,embedding_d,embedding_u,times)):
            with torch.autocast(source.device.type,enabled=False):
                value=value*(road[index]+1e-7)
                degree=value.sum(-1,keepdim=True)
                result.extend(graph_orders(value/torch.where(degree>0,degree,torch.ones_like(degree)),self.order,mask))
        return result


class D2STGNN(nn.Module):
    def __init__(self,metadata,input_window,output_window,num_hidden=32,node_dim=10,forecast_dim=256,output_hidden=512,
                 k_s=2,k_t=3,num_layers=5,time_emb_dim=10,gap=3,use_pre=False,dy_graph=True,sta_graph=True,
                 use_time_of_day=True,use_day_of_week=True,attention_heads=4,dropout=.1,steps_per_day=0,dynamic_channel=0):
        super().__init__()
        for name,value in dict(input_window=input_window,output_window=output_window,num_hidden=num_hidden,node_dim=node_dim,
                forecast_dim=forecast_dim,output_hidden=output_hidden,k_s=k_s,k_t=k_t,num_layers=num_layers,
                time_emb_dim=time_emb_dim,gap=gap,attention_heads=attention_heads).items():
            if type(value) is not int or value<1: raise ValueError('D2STGNN '+name+' must be a positive integer')
        if num_hidden%attention_heads or input_window<1+num_layers*(k_t-1):
            raise ValueError('D2STGNN requires divisible attention heads and enough history for all temporal layers')
        if type(steps_per_day) is not int or steps_per_day<0 or type(dynamic_channel) is not int or dynamic_channel<0 or not math.isfinite(dropout) or not 0<=dropout<1:
            raise ValueError('D2STGNN time slots, channel or dropout are invalid')
        if any(type(value) is not bool for value in (use_pre,dy_graph,sta_graph,use_time_of_day,use_day_of_week)):
            raise ValueError('D2STGNN graph and calendar switches must be boolean')
        if metadata.input_layout.get('kind')!='window': raise ValueError('D2STGNN requires continuous window metadata')
        self.steps,self.horizon,self.nodes,self.features,self.outputs=input_window,output_window,metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.tid,self.dow=calendar_metadata(metadata,use_time_of_day,use_day_of_week)
        calendar=metadata.input_layout.get('calendar',{});calendar_columns={*calendar.get('day_of_week',())}
        if calendar.get('time_in_day') is not None: calendar_columns.add(calendar['time_in_day'])
        self.numeric=tuple(i for i in range(self.features) if i not in calendar_columns)
        if not self.numeric or dynamic_channel>=len(self.numeric): raise ValueError('D2STGNN dynamic_channel must select a numeric history column')
        self.dynamic_channel=dynamic_channel;self.use_pre,self.dy_graph,self.sta_graph=use_pre,dy_graph,sta_graph
        self.k_s,self.gap=k_s,min(gap,output_window);chunks=math.ceil(output_window/self.gap)
        self.slots=steps_in_day(metadata,steps_per_day) if use_time_of_day else 0
        self.T_i_D_emb=nn.Parameter(torch.empty(self.slots,time_emb_dim)) if use_time_of_day else None
        self.D_i_W_emb=nn.Parameter(torch.empty(7,time_emb_dim)) if use_day_of_week else None
        self.node_emb_u=nn.Parameter(torch.empty(self.nodes,node_dim));self.node_emb_d=nn.Parameter(torch.empty(self.nodes,node_dim))
        for value in (self.node_emb_u,self.node_emb_d,self.T_i_D_emb,self.D_i_W_emb):
            if value is not None: nn.init.xavier_uniform_(value)
        self.register_buffer('diagonal_mask',~torch.eye(self.nodes,dtype=torch.bool))
        road=transition_supports(metadata.adjacency,self.nodes) if use_pre or dy_graph else torch.empty(0,self.nodes,self.nodes)
        self.register_buffer('road_supports',road)
        predefined=[term for graph in road for term in graph_orders(graph,k_s,self.diagonal_mask)] if use_pre else []
        self.register_buffer('predefined_orders',torch.stack(predefined) if predefined else torch.empty(0,self.nodes,self.nodes))
        time_dim=time_emb_dim*(int(use_time_of_day)+int(use_day_of_week));support_count=k_s*(2*int(use_pre)+2*int(dy_graph)+int(sta_graph))
        if dy_graph: self.dynamic_graph_constructor=DynamicGraphConstructor(input_window,num_hidden,node_dim,time_dim,dropout,k_s)
        self.embedding=nn.Linear(len(self.numeric),num_hidden)
        self.layers=nn.ModuleList(DecoupleLayer(num_hidden,forecast_dim,chunks,k_t,support_count,2*node_dim+time_dim,
            attention_heads,dropout,input_window+chunks,index==num_layers-1) for index in range(num_layers))
        self.out_fc_1=nn.Linear(forecast_dim,output_hidden);self.out_fc_2=nn.Linear(output_hidden,self.gap*self.outputs)

    def prepare_inputs(self,source):
        times=[]
        if self.T_i_D_emb is not None:
            clock=source[...,self.tid];clock=clock.float() if clock.dtype in (torch.float16,torch.bfloat16) else clock
            index=(clock*self.slots+torch.finfo(clock.dtype).eps*self.slots).floor().long().clamp(0,self.slots-1)
            times.append(self.T_i_D_emb[index])
        if self.D_i_W_emb is not None: times.append(self.D_i_W_emb[source[...,self.dow].argmax(-1)])
        return source[...,self.numeric],times

    def forward(self,batch):
        source=batch['X']
        if source.ndim!=4 or source.shape[1:]!=(self.steps,self.nodes,self.features):
            raise ValueError('D2STGNN X must match [B, history, nodes, features]')
        numeric,times=self.prepare_inputs(source)
        supports=list(self.predefined_orders)
        if self.dy_graph: supports+=self.dynamic_graph_constructor(numeric[...,self.dynamic_channel],self.node_emb_d,self.node_emb_u,times,self.road_supports,self.diagonal_mask)
        if self.sta_graph:
            with torch.autocast(source.device.type,enabled=False):
                graph=(self.node_emb_u@self.node_emb_d.T).relu().softmax(-1)
                supports+=graph_orders(graph,self.k_s,self.diagonal_mask)
        embeddings=torch.cat([*times,self.node_emb_u[None,None].expand(len(source),self.steps,-1,-1),
            self.node_emb_d[None,None].expand(len(source),self.steps,-1,-1)],-1)
        remaining=self.embedding(numeric);forecast=None
        for layer in self.layers:
            remaining,diffusion,inherent=layer(remaining,supports,embeddings)
            current=diffusion+inherent
            forecast=current if forecast is None else forecast+current
        prediction=self.out_fc_2(self.out_fc_1(forecast.relu()).relu())
        prediction=prediction.unflatten(-1,(self.gap,self.outputs)).transpose(1,2).flatten(2,3).transpose(1,2)
        return prediction[:,:self.horizon]
