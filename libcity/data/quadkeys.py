"""Mercator quadkeys and the small vocabulary GeoSAN actually needs."""
from collections import Counter
import math

import torch
from torch.nn.utils.rnn import pad_sequence


def quadkey(longitude,latitude,level=17):
    latitude = min(max(latitude,-85.05112878),85.05112878)
    sine = math.sin(math.radians(latitude))
    x = (longitude+180)/360
    y = .5-math.log((1+sine)/(1-sine))/(4*math.pi)
    size = 256<<level
    tile_x = int(min(max(x*size+.5,0),size-1))//256
    tile_y = int(min(max(y*size+.5,0),size-1))//256
    return ''.join(str(int(bool(tile_x&(1<<bit)))+2*int(bool(tile_y&(1<<bit)))) for bit in reversed(range(level)))


def quadkey_ngrams(value,width=6):
    return [value[start:start+width] for start in range(len(value)-width+1)]


class QuadkeyVocabulary:
    """Field(unk_token=None): PAD=0, frequency descending, lexical tie order."""
    def build_vocab(self,sequences,min_freq=1):
        self.freqs = Counter(token for sequence in sequences for token in sequence)
        self.itos = ['<pad>',*[token for token,count in sorted(self.freqs.items(),key=lambda item:(-item[1],item[0]))
            if token!='<pad>' and count>=max(1,min_freq)]]
        self.stoi = {token:index for index,token in enumerate(self.itos)}

    @property
    def vocab(self):
        return self

    def numericalize(self,sequences):
        rows = [torch.tensor([self.stoi[token] for token in sequence],dtype=torch.long) for sequence in sequences]
        return pad_sequence(rows,batch_first=True) if rows else torch.empty(0,0,dtype=torch.long)
