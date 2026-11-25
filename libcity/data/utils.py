from libcity.registry import load_component
import numpy as np
from torch.utils.data import DataLoader
import copy

from libcity.data.list_dataset import ListDataset
from libcity.data.batch import Batch, BatchPAD


def get_dataset(config):
    dataset = load_component('dataset_class', config['dataset_class'])
    return dataset(config)


def split_counts(num_samples, train_rate, eval_rate):
    if (not 0 < train_rate < 1 or not 0 < eval_rate < 1
            or train_rate + eval_rate >= 1):
        raise ValueError('train_rate and eval_rate must be positive, with their sum below 1.')
    num_train = round(num_samples * train_rate)
    num_test = round(num_samples * (1 - train_rate - eval_rate))
    num_val = num_samples - num_train - num_test
    if min(num_train, num_val, num_test) <= 0:
        raise ValueError(
            f'{num_samples} window samples produce empty train/validation/test splits '
            f'({num_train}, {num_val}, {num_test}). Add data, shorten the windows, '
            'or adjust train_rate/eval_rate.')
    return num_train, num_val, num_test


class BatchCollator:
    """A spawn-picklable collator that only copies fields mutated in-place."""

    def __init__(self, feature_name, pad_item=None, pad_max_len=None, padded=False):
        self.feature_name = feature_name
        self.pad_item = pad_item
        self.pad_max_len = pad_max_len
        self.padded = padded or pad_item is not None

    def __call__(self, samples):
        if not self.padded:
            batch = Batch(self.feature_name)
            for item in samples:
                if len(item) != len(self.feature_name):
                    raise KeyError('sample does not match feature_name')
                batch.append(item)
        else:
            batch = BatchPAD(self.feature_name, self.pad_item, self.pad_max_len)
            for item in samples:
                if len(item) != len(self.feature_name):
                    raise KeyError('sample does not match feature_name')
                # Ragged/opaque features can be changed by BatchPAD.to_tensor.
                # Dense int/float features are stacked into new arrays instead.
                values = [value if kind in ('int', 'float') else copy.deepcopy(value)
                          for value, kind in zip(item, self.feature_name.values())]
                batch.append(values)
            batch.padding()
        # Own dense storage before handing the batch to consumers. This stacks
        # each feature once, rather than copying every sample and stacking again.
        for key, kind in self.feature_name.items():
            if kind in ('int', 'float'):
                batch.data[key] = np.asarray(batch.data[key])
        return batch


def _pad_with_last_sample(samples, batch_size):
    """Pad heterogeneous (X, y) samples without NumPy's object-array coercion."""
    values = list(samples)
    if values:
        values.extend([values[-1]] * ((-len(values)) % batch_size))
    return values


def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=True,
                        pad_with_last_sample=False):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): Only pad training batches by repeating the last
            sample. Validation/test keep their real final batch and sample count.

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    if pad_with_last_sample:
        train_data = _pad_with_last_sample(train_data, batch_size)

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    collator = BatchCollator(feature_name)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    return train_dataloader, eval_dataloader, test_dataloader


def generate_dataloader_pad(train_data, eval_data, test_data, feature_name,
                            batch_size, num_workers, pad_item=None,
                            pad_max_len=None, shuffle=True):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        pad_item(dict): 用于将不定长的特征补齐到一样的长度，每个特征名作为 key，若某特征名不在该 dict 内则不进行补齐。
        pad_max_len(dict): 用于截取不定长的特征，对于过长的特征进行剪切
        shuffle(bool): shuffle

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    collator = BatchCollator(feature_name, pad_item, pad_max_len, padded=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    return train_dataloader, eval_dataloader, test_dataloader
