"""Tensor dictionary loaders for retained dataset preparation code.

DataBundle adapters own experiments; these helpers select no models or executors.
"""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from libcity.data.list_dataset import ListDataset


def split_counts(num_samples,train_rate,eval_rate):
    if not 0<train_rate<1 or not 0<eval_rate<1 or train_rate+eval_rate>=1:
        raise ValueError('train_rate and eval_rate must be positive, with their sum below 1.')
    train=round(num_samples*train_rate)
    test=round(num_samples*(1-train_rate-eval_rate))
    val=num_samples-train-test
    if min(train,val,test)<=0:
        raise ValueError('Window samples produce empty train/validation/test splits')
    return train,val,test


class TensorCollator:
    """Spawn-picklable owned tensors and explicit lengths for padded sequences."""
    def __init__(self,features,pad_item=None,pad_max_len=None):
        self.features=features
        self.pad_item=pad_item or {}
        self.pad_max_len=pad_max_len or {}

    def __call__(self,samples):
        if any(len(sample)!=len(self.features) for sample in samples):
            raise ValueError('Sample fields do not match feature declarations')
        result={}
        for index,(name,kind) in enumerate(self.features.items()):
            dtype=torch.float32 if kind in ('float','array of float') else torch.long
            values=[torch.as_tensor(np.asarray(sample[index]),dtype=dtype) for sample in samples]
            if name in self.pad_item:
                maximum=self.pad_max_len.get(name)
                if maximum is not None:
                    values=[value[-maximum:] for value in values]
                result[name+'_lengths']=torch.tensor([len(value) for value in values])
                result[name]=pad_sequence(values,batch_first=True,padding_value=self.pad_item[name])
            else:
                result[name]=torch.stack(values)
        return result


def _pad_with_last_sample(samples,batch_size):
    values=list(samples)
    if values:
        values.extend([values[-1]]*((-len(values))%batch_size))
    return values


def _loaders(parts,collator,batch_size,num_workers,shuffle):
    return tuple(DataLoader(ListDataset(values),batch_size=batch_size,num_workers=num_workers,
        collate_fn=collator,shuffle=shuffle if index==0 else False) for index,values in enumerate(parts))


def generate_dataloader(train_data,eval_data,test_data,feature_name,batch_size,num_workers,
                        shuffle=True,pad_with_last_sample=False):
    train=_pad_with_last_sample(train_data,batch_size) if pad_with_last_sample else train_data
    return _loaders((train,eval_data,test_data),TensorCollator(feature_name),batch_size,num_workers,shuffle)


def generate_dataloader_pad(train_data,eval_data,test_data,feature_name,batch_size,num_workers,
                            pad_item=None,pad_max_len=None,shuffle=True):
    return _loaders((train_data,eval_data,test_data),TensorCollator(feature_name,pad_item,pad_max_len),
                    batch_size,num_workers,shuffle)
