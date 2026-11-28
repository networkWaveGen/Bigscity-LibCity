"""Local flow gates and periodically shifted attention without future labels."""
import torch
from torch import nn


def conv_bank(steps,channels,sigmoid=False):
    return nn.ModuleList([nn.Sequential(nn.Conv2d(channels,64,3,padding=1),nn.ReLU(),
        *([nn.Sigmoid()] if sigmoid else [])) for _ in range(steps)])


class CBAAttention(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.Wq,self.Wh,self.v=nn.Linear(hidden,hidden),nn.Linear(hidden,hidden),nn.Linear(hidden,1)
        for layer in (self.Wq,self.Wh,self.v):
            nn.init.xavier_uniform_(layer.weight,nn.init.calculate_gain('tanh'))

    def forward(self,memory,query):
        scores=self.v(torch.tanh(self.Wh(memory)+self.Wq(query)[:,None])).squeeze(-1).softmax(-1)
        return (memory*scores[...,None]).sum(1)


class STDN(nn.Module):
    def __init__(self,metadata,input_window,output_window,cnn_flat_size=128,lstm_out_size=128):
        super().__init__()
        for name,value in (('input_window',input_window),('output_window',output_window),('cnn_flat_size',cnn_flat_size),('lstm_out_size',lstm_out_size)):
            if type(value) is not int or value<1: raise ValueError('STDN '+name+' must be a positive integer')
        layout=metadata.input_layout
        if layout.get('kind')!='stdn_grid' or layout['length']!=input_window:
            raise ValueError('STDN requires matching stdn_grid history metadata')
        self.steps,self.horizon=input_window,output_window
        self.periods,self.period_steps=layout['periods'],layout['period_steps']
        self.nodes,self.features,self.outputs=metadata.num_nodes,metadata.feature_dim,metadata.output_dim
        self.patch=2*layout['cnn_nbhd_size']+1
        self.vector=layout['feature_vector_width'];self.flow_width=4*self.patch**2
        for stage in ('1st','2nd','3rd'):
            setattr(self,'nbhd_cnns_'+stage,conv_bank(input_window,self.features if stage=='1st' else 64))
            # Preserve the author's independent raw-flow gates for recent steps.
            setattr(self,'flow_cnns_'+stage,conv_bank(input_window,4,True))
        dense=lambda:nn.Sequential(nn.Flatten(),nn.Linear(64*self.patch**2,cnn_flat_size),nn.ReLU())
        self.nbhd_vecs=nn.ModuleList([dense() for _ in range(input_window)])
        self.lstm=nn.LSTM(self.vector+cnn_flat_size,lstm_out_size,batch_first=True)
        for stage in ('1st','2nd','3rd'):
            setattr(self,'att_nbhd_cnns_'+stage,nn.ModuleList([conv_bank(self.period_steps,self.features if stage=='1st' else 64) for _ in range(self.periods)]))
            setattr(self,'att_flow_cnns_'+stage,nn.ModuleList([conv_bank(self.period_steps,4 if stage=='1st' else 64) for _ in range(self.periods)]))
        self.att_nbhd_vecs=nn.ModuleList([nn.ModuleList([dense() for _ in range(self.period_steps)]) for _ in range(self.periods)])
        self.att_lstms=nn.ModuleList([nn.LSTM(self.vector+cnn_flat_size,lstm_out_size,batch_first=True) for _ in range(self.periods)])
        self.att_low_level=nn.ModuleList([CBAAttention(lstm_out_size) for _ in range(self.periods)])
        self.att_high_level=nn.LSTM(lstm_out_size,lstm_out_size,batch_first=True)
        self.lstm_all=nn.Linear(2*lstm_out_size,output_window*self.outputs)

    def spatial(self,volume,flow,period=None):
        outputs=[]
        for index in range(volume.shape[1]):
            values,raw_flow=volume[:,index],flow[:,index]
            flow_values=raw_flow
            for stage in ('1st','2nd','3rd'):
                if period is None:
                    values=getattr(self,'nbhd_cnns_'+stage)[index](values)
                    gate=getattr(self,'flow_cnns_'+stage)[index](raw_flow)
                else:
                    values=getattr(self,'att_nbhd_cnns_'+stage)[period][index](values)
                    flow_values=getattr(self,'att_flow_cnns_'+stage)[period][index](flow_values)
                    gate=flow_values.sigmoid()
                values=values*gate
            dense=self.nbhd_vecs[index] if period is None else self.att_nbhd_vecs[period][index]
            outputs.append(dense(values))
        return torch.stack(outputs,dim=1)

    def forward(self,batch):
        observed,context=batch['X'],batch['history_features']
        total=self.steps+self.periods*self.period_steps
        if observed.ndim!=6 or observed.shape[1:]!=(total,self.nodes,self.patch,self.patch,self.features):
            raise ValueError('STDN X must match [B, sampled history, nodes, patch rows, patch columns, observations]')
        size=len(observed)
        if context.shape!=(size,total,self.nodes,self.flow_width+self.vector):
            raise ValueError('STDN history_features must contain aligned flow patches and volume vectors')
        volume=observed.permute(0,2,1,5,3,4).reshape(size*self.nodes,total,self.features,self.patch,self.patch)
        context=context.transpose(1,2).reshape(size*self.nodes,total,-1)
        flow=context[...,:self.flow_width].reshape(size*self.nodes,total,self.patch,self.patch,4).permute(0,1,4,2,3)
        vector=context[...,self.flow_width:]
        recent=self.spatial(volume[:,:self.steps],flow[:,:self.steps])
        query=self.lstm(torch.cat((vector[:,:self.steps],recent),dim=-1))[1][0][-1]
        attended=[]
        for period in range(self.periods):
            selection=slice(self.steps+period*self.period_steps,self.steps+(period+1)*self.period_steps)
            spatial=self.spatial(volume[:,selection],flow[:,selection],period)
            memory=self.att_lstms[period](torch.cat((vector[:,selection],spatial),dim=-1))[0]
            attended.append(self.att_low_level[period](memory,query))
        periodic=self.att_high_level(torch.stack(attended,dim=1))[1][0][-1]
        prediction=self.lstm_all(torch.cat((periodic,query),dim=-1)).tanh()
        return prediction.reshape(size,self.nodes,self.horizon,self.outputs).transpose(1,2)
