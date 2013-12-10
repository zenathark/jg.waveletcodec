
from __future__ import division
import numpy as np
import waveletcodec.tools as tls


class integDCT(object):

    one_ker = np.array([
        [1, 1, 1, 1],
        [2, 1, -1, -2],
        [1, -1, -1, 1],
        [1, -2, 2, -1]
    ])
    i_one_ker = np.array([
        [1, 1, 1, 1],
        [1, 1 / 2, -1 / 2, -1],
        [1, -1, -1, 1],
        [1 / 2, -1, 1, -1 / 2]
    ])
    norm_one_ker = []
    norm_i_one_ker = []
    qs = np.arange(0.625, 224, 0.625 / 6)

    def __init__(self):
        r = 1 / np.sqrt((self.one_ker ** 2).sum(1))
        r = np.array([r, r, r, r]).T
        self.norm_one_ker = self.one_ker * r
        ri = 1 / np.sqrt((self.i_one_ker ** 2).sum(1))
        ri = np.array([ri, ri, ri, ri]).T
        self.norm_i_one_ker = self.i_one_ker * ri

    def dct4x4(self, data, q):
        data = tls.zero_padding_n(data, 4, False)
        new_data = np.zeros(data.shape)
        for i in range(0, data.shape[0], 4):
            for j in range(0, data.shape[1], 4):
                new_data[i:i + 4, j:j + 4] = self._dct4x4(
                    data[i:i + 4, j:j + 4], q)
        return new_data

    def idct4x4(self, data, q, shape):
        new_data = np.zeros(data.shape)
        for i in range(0, data.shape[0], 4):
            for j in range(0, data.shape[1], 4):
                new_data[i:i + 4, j:j + 4] = self._idct4x4(
                    data[i:i + 4, j:j + 4], q)
        new_data = tls.unpadding(new_data, shape)
        return new_data

    def _dct4x4(self, data, i):
        t = self.norm_one_ker.dot(data).dot(self.norm_one_ker.T)
        t = tls.quantize(t, self.qs[i], dtype=np.int)
        return t

    def _idct4x4(self, data, i):
        t = self.norm_i_one_ker.T.dot(data).dot(self.norm_i_one_ker)
        t = tls.quantize(t, 1 / self.qs[i], dtype=np.int)
        return t
