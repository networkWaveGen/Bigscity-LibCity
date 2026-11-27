"""SERM with intact pretrained semantics and tensor-only variable-length inputs."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SERM(nn.Module):
    def __init__(self,*,metadata,loc_emb_size=50,tim_emb_size=50,hidden_size=50):
        super().__init__()
        if min(loc_emb_size,tim_emb_size,hidden_size)<1:
            raise ValueError('SERM requires positive embedding and hidden widths')
        info = metadata.info
        self.loc_pad,self.tim_pad = info['loc_pad'],info['tim_pad']
        vectors = torch.tensor(info['word_vectors'].copy(),dtype=torch.float32)
        tokens = torch.tensor(info['poi_word_ids'].copy(),dtype=torch.long)
        if (vectors.ndim!=2 or vectors.shape[0]<1 or vectors.shape[1]<1 or not torch.isfinite(vectors).all()
                or tokens.ndim!=2 or tokens.shape[0]!=info['loc_size'] or tokens.shape[1]<1
                or (tokens<0).any() or (tokens>=len(vectors)).any()):
            raise ValueError('SERM requires aligned POI word IDs and finite pretrained vectors')
        self.register_buffer('poi_words',tokens)
        self.emb_loc = nn.Embedding(info['loc_size'],loc_emb_size,padding_idx=self.loc_pad)
        self.emb_tim = nn.Embedding(info['tim_size'],tim_emb_size,padding_idx=self.tim_pad)
        self.emb_user = nn.Embedding(info['uid_size'],info['loc_size'])
        self.emb_text = nn.EmbeddingBag.from_pretrained(vectors,freeze=False,mode='mean',padding_idx=0)
        self.lstm = nn.LSTM(loc_emb_size+tim_emb_size+vectors.shape[1],hidden_size,batch_first=True)
        self.dense = nn.Linear(hidden_size,info['loc_size'])
        # Initialize ordinary layers only: pretrained semantic vectors stay intact.
        for embedding in (self.emb_loc,self.emb_tim,self.emb_user):
            nn.init.xavier_normal_(embedding.weight)
        with torch.no_grad():
            self.emb_loc.weight[self.loc_pad].zero_()
            self.emb_tim.weight[self.tim_pad].zero_()
        nn.init.xavier_uniform_(self.dense.weight)
        for name,parameter in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(parameter)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(parameter)
            elif 'bias' in name:
                nn.init.zeros_(parameter)

    def semantics(self,locations):
        tokens = self.poi_words[locations]
        return self.emb_text(tokens.reshape(-1,tokens.shape[-1])).reshape(*locations.shape,-1)

    def forward(self,batch):
        locations,times,lengths,users = (batch[key] for key in ('current_loc','current_tim','lengths','uid'))
        if (locations.ndim!=2 or times.shape!=locations.shape or lengths.shape!=(len(locations),)
                or users.shape!=(len(locations),) or (lengths<1).any() or (lengths>locations.shape[1]).any()):
            raise ValueError('SERM expects matching padded locations/times, users and valid lengths')
        stop = int(lengths.max())
        locations,times = locations[:,:stop],times[:,:stop]
        valid = torch.arange(stop,device=locations.device)[None]<lengths[:,None]
        locations,times = locations.masked_fill(~valid,self.loc_pad),times.masked_fill(~valid,self.tim_pad)
        if ((locations[valid]<1).any() or (locations[valid]>=self.emb_loc.num_embeddings).any()
                or (times[valid]<0).any() or (times[valid]>=self.tim_pad).any()
                or (users<0).any() or (users>=self.emb_user.num_embeddings).any()):
            raise ValueError('SERM requires valid observed locations, calendar times and user IDs')
        embedded = torch.cat((self.emb_loc(locations),self.emb_tim(times),self.semantics(locations)),dim=-1)
        packed = pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(hidden,_) = self.lstm(packed)
        logits = self.dense(hidden[-1])+self.emb_user(users)
        return torch.log_softmax(logits[:,1:],dim=-1)
