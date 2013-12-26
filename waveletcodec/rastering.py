"""Rastering algorithms.

.. module::rastering
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""
from __future__ import division
import math
import numpy as np


def get_z_order(dim):
    mtx = []
    n = int(math.log(dim, 2))
    pows = range(int(n / 2))
    for i in range(dim):
        x = 0
        y = 0
        for j in pows:
            x |= ((i >> 2 * j) & 1) << j
            y |= ((i >> 2 * j + 1) & 1) << j
        mtx += [vector((y, x))]
    return mtx


def get_z_index(v):
    j, i = v
    j = int(j)
    i = int(i)
    z = (((j * 0x0101010101010101 & 0x8040201008040201) *
          0x0102040810204081 >> 49) & 0x5555 |
         ((i * 0x0101010101010101 & 0x8040201008040201) *
          0x0102040810204081 >> 48) & 0xAAAA)
    return z


class vector(object):
    def __init__(self, data, entry_type="-"):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.entry_type = type
        self.deleted = False

    def __hash__(self):
        return hash(tuple(self.data))

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return vector(self.data + other.data)
        else:
            return vector(self.data + np.array(other))

    def __eq__(self, other):
        if (self.__hash__() == other.__hash__()):
            return True
        else:
            return False

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    def __mul__(self, other):
        return vector(self.data * other)

    def __rmul__(self, other):
        return vector(self.data * other)

    def __lt__(self, other):
        if (isinstance(other, np.ndarray)):
            return np.all(self.data < other)
        else:
            return np.all(self.data < np.array(other))

    def tolist(self):
        return self.data.tolist()

    def __getitem__(self, index):
        return self.data[index]
