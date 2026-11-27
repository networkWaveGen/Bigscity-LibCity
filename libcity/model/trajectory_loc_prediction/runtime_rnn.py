"""Packed variable-length RNN with full-vocabulary next-location scores."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self,*,metadata,loc_emb_size=32,tim_emb_size=8,hidden_size=64,rnn_type='GRU',dropout_p=.1):
        super().__init__()
        info=metadata.info
        if rnn_type not in ('RNN','LSTM','GRU'):
            raise ValueError('rnn_type must be RNN/LSTM/GRU')
        self.emb_loc=nn.Embedding(info['loc_size'],loc_emb_size,padding_idx=info['loc_pad'])
        self.emb_tim=nn.Embedding(info['tim_size'],tim_emb_size,padding_idx=info['tim_pad'])
        self.rnn=getattr(nn,rnn_type)(loc_emb_size+tim_emb_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,metadata.num_nodes)
        self.dropout=nn.Dropout(dropout_p)

    def forward(self,batch):
        source=self.dropout(torch.cat((self.emb_loc(batch['current_loc']),self.emb_tim(batch['current_tim'])),dim=-1))
        packed=pack_padded_sequence(source,batch['lengths'].cpu(),batch_first=True,enforce_sorted=False)
        output,_=self.rnn(packed)
        output,_=pad_packed_sequence(output,batch_first=True)
        final=output[torch.arange(len(output),device=output.device),batch['lengths']-1]
        return torch.log_softmax(self.fc(self.dropout(torch.selu(final))),dim=-1)
