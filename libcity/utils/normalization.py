import numpy as np


def _nonzero_scale(value):
    """Use unit scale for constant data, preserving an invertible transform."""
    scale = np.asarray(value)
    safe = np.where(scale == 0, 1, scale)
    return safe.item() if safe.ndim == 0 else safe


class Scaler:
    """
    归一化接口
    """

    def transform(self, data):
        """
        数据归一化接口

        Args:
            data(np.ndarray): 归一化前的数据

        Returns:
            np.ndarray: 归一化后的数据
        """
        raise NotImplementedError("Transform not implemented")

    def inverse_transform(self, data):
        """
        数据逆归一化接口

        Args:
            data(np.ndarray): 归一化后的数据

        Returns:
            np.ndarray: 归一化前的数据
        """
        raise NotImplementedError("Inverse_transform not implemented")


class NoneScaler(Scaler):
    """
    不归一化
    """

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class NormalScaler(Scaler):
    """
    除以最大值归一化
    x = x / x.max
    """

    def __init__(self, maxx):
        self.max = maxx
        self.scale = _nonzero_scale(maxx)

    def transform(self, data):
        return data / self.scale

    def inverse_transform(self, data):
        return data * self.scale


class StandardScaler(Scaler):
    """
    Z-score归一化
    x = (x - x.mean) / x.std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.scale = _nonzero_scale(std)

    def transform(self, data):
        return (data - self.mean) / self.scale

    def inverse_transform(self, data):
        return (data * self.scale) + self.mean


class MinMax01Scaler(Scaler):
    """
    MinMax归一化 结果区间[0, 1]
    x = (x - min) / (max - min)
    """

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx
        self.scale = _nonzero_scale(maxx - minn)

    def transform(self, data):
        return (data - self.min) / self.scale

    def inverse_transform(self, data):
        return data * self.scale + self.min


class MinMax11Scaler(Scaler):
    """
    MinMax归一化 结果区间[-1, 1]
    x = (x - min) / (max - min)
    x = x * 2 - 1
    """

    def __init__(self, minn, maxx):
        self.min = minn
        self.max = maxx
        self.scale = _nonzero_scale(maxx - minn)

    def transform(self, data):
        return ((data - self.min) / self.scale) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * self.scale + self.min


class LogScaler(Scaler):
    """
    Log scaler
    x = log(x+eps)
    """

    def __init__(self, eps=0.999):
        self.eps = eps

    def transform(self, data):
        return np.log(data + self.eps)

    def inverse_transform(self, data):
        return np.exp(data) - self.eps
