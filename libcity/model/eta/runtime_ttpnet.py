"""TTPNet's speed, road and bidirectional route networks without task state."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


def recurrent_sequence(module,values,lengths):
    packed = pack_padded_sequence(values,lengths.cpu(),batch_first=True,enforce_sorted=False)
    packed,_ = module(packed)
    return pad_packed_sequence(packed,batch_first=True,total_length=values.shape[1])[0]


class ShortSpeed(nn.Module):
    def __init__(self):
        super().__init__()
        self.short_cnn = nn.Conv1d(3,4,2)
        self.short_rnn = nn.RNN(4,16,batch_first=True)

    def forward(self,values):
        # One shared CNN/RNN over all route points and three spatial branches.
        shape = values.shape
        values = values.reshape(-1,4,3)
        values = self.short_cnn(values.transpose(1,2)).tanh().transpose(1,2)
        values,_ = self.short_rnn(values)
        return values[:,-1].reshape(shape[0],shape[1],48)


class LongSpeed(nn.Module):
    def __init__(self):
        super().__init__()
        self.long_cnn = nn.Conv1d(3,4,3)
        self.long_rnn = nn.RNN(4,16,batch_first=True)

    def forward(self,values):
        shape = values.shape
        values = self.long_cnn(values.reshape(-1,7,3).transpose(1,2)).transpose(1,2)
        values,_ = self.long_rnn(values)
        return values[:,-1].reshape(shape[0],shape[1],16)


class SpeedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.shortspeed_net,self.longspeed_net = ShortSpeed(),LongSpeed()
        self.process_speeds = nn.Linear(64,32)
        self.speed_lstm = nn.LSTM(32,32,batch_first=True)

    def forward(self,short,long,lengths):
        values = self.process_speeds(torch.cat((self.shortspeed_net(short),self.longspeed_net(long)),dim=-1)).tanh()
        return recurrent_sequence(self.speed_lstm,values,lengths)


class Road(nn.Module):
    def __init__(self,embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings.copy()),freeze=False)
        self.process_coords = nn.Linear(2+embeddings.shape[1],32)

    def forward(self,coords,ids):
        return self.process_coords(torch.cat((coords,self.embedding(ids)),dim=-1)).tanh()


class RoadLSTM(nn.Module):
    def __init__(self,embeddings):
        super().__init__()
        self.Road_net = Road(embeddings)
        self.Road_lstm = nn.LSTM(32,32,batch_first=True)

    def forward(self,coords,ids,lengths):
        return recurrent_sequence(self.Road_lstm,self.Road_net(coords,ids),lengths)


class Attr(nn.Module):
    def __init__(self,count,uid_width,week_width,time_width):
        super().__init__()
        self.uid_em = nn.Embedding(count,uid_width)
        self.weekid_em = nn.Embedding(7,week_width)
        self.timeid_em = nn.Embedding(96,time_width)

    def forward(self,batch):
        return torch.cat((self.uid_em(batch['uid']),self.weekid_em(batch['weekid']),self.timeid_em(batch['timeid']),
            batch['dist'],batch['holiday'].to(batch['dist'].dtype)),dim=-1)


class PredictionBiLSTM(nn.Module):
    def __init__(self,info,uid_width,week_width,time_width,dropout):
        super().__init__()
        self.attr_net = Attr(info['uid_size'],uid_width,week_width,time_width)
        self.speed_lstm = SpeedLSTM()
        self.road_lstm = RoadLSTM(info['road_embeddings'])
        width = uid_width+week_width+time_width+2+64
        self.lnhiddens = nn.LayerNorm(width)
        self.bi_lstm = nn.LSTM(width,64,2,batch_first=True,bidirectional=True,dropout=dropout)

    def forward(self,batch):
        lengths = batch['lengths']
        speed = self.speed_lstm(batch['short_traffic'],batch['long_traffic'],lengths)
        road = self.road_lstm(batch['coords'],batch['road_id'],lengths)
        attributes = self.attr_net(batch)[:,None].expand(-1,road.shape[1],-1)
        values = self.lnhiddens(torch.cat((attributes,speed,road),dim=-1))
        return recurrent_sequence(self.bi_lstm,values,lengths)


class TTPNet(nn.Module):
    def __init__(self,*,metadata,uid_emb_size=8,weekid_emb_size=3,timid_emb_size=8,dropout=.25):
        super().__init__()
        if min(uid_emb_size,weekid_emb_size,timid_emb_size)<1 or not 0<=dropout<1:
            raise ValueError('TTPNet requires positive embedding widths and dropout in [0,1)')
        self.bi_lstm = PredictionBiLSTM(metadata.info,uid_emb_size,weekid_emb_size,timid_emb_size,dropout)
        self.input2hid,self.hid2hid,self.hid2out = nn.Linear(128,128),nn.Linear(128,64),nn.Linear(64,1)

    def cumulative(self,batch):
        coords,lengths = batch['coords'],batch['lengths']
        if (coords.ndim!=3 or coords.shape[-1]!=2 or lengths.shape!=(len(coords),) or not len(coords)
                or (lengths<2).any() or (lengths>coords.shape[1]).any()):
            raise ValueError('TTPNet needs matching route coordinates and lengths of at least two')
        count,maximum = len(coords),int(lengths.max())
        expected = dict(road_id=coords.shape[:2],short_traffic=(*coords.shape[:2],3,4,3),long_traffic=(*coords.shape[:2],7,3),
            uid=(count,),weekid=(count,),timeid=(count,),dist=(count,1),holiday=(count,1))
        if any(batch[name].shape!=shape for name,shape in expected.items()):
            raise ValueError('TTPNet route and departure attribute shapes do not match')
        valid = torch.arange(maximum,device=coords.device)[None]<lengths[:,None]
        clean = {name:batch[name] for name in ('uid','weekid','timeid','dist','holiday','lengths')}
        for name in ('coords','road_id','short_traffic','long_traffic'):
            values = batch[name][:,:maximum]
            mask = valid.reshape(*valid.shape,*([1]*(values.ndim-2)))
            clean[name] = values.masked_fill(~mask,0)
            if not torch.isfinite(clean[name]).all():
                raise ValueError('TTPNet valid route features must be finite')
        if ((clean['road_id']<0).any() or (clean['road_id']>=self.bi_lstm.road_lstm.Road_net.embedding.num_embeddings).any()
                or any((clean[name]<0).any() or (clean[name]>=limit).any() for name,limit in
                    (('uid',self.bi_lstm.attr_net.uid_em.num_embeddings),('weekid',7),('timeid',96)))
                or not torch.isfinite(clean['dist']).all() or not torch.isfinite(clean['holiday']).all()
                or ((clean['holiday']!=0)&(clean['holiday']!=1)).any()):
            raise ValueError('TTPNet road IDs and departure attributes are invalid')
        hidden = self.bi_lstm(clean)
        pooled = hidden.cumsum(1)[:,1:]
        values = self.hid2out(F.relu(self.hid2hid(F.relu(self.input2hid(pooled))))).squeeze(-1)
        valid = valid[:,1:]
        final = values.gather(1,(lengths-2)[:,None])
        return final,values,valid

    def forward(self,batch):
        return self.cumulative(batch)[0]

    def forward_with_aux(self,batch):
        return self.cumulative(batch)
