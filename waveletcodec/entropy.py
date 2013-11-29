"""Module for Entropy Coding Algorithms.

.. module::entropy
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

from __future__ import division
import itertools as it


class arithmeticb(object):

    """Class of a binary arithmetic codec implemented assuming infinite
    floating point presicion

    """

    _ctr = 0
    _l = 0
    _h = 1
    _buff = 0
    _output = []
    _p = []

    def __init__(self):
        super(arithmeticb, self).__init__()

    def _initialize(self, data):
        self._ctr = 0
        self._l = 0
        self._h = 0.9999
        self._buff = 0
        self._output = []
        #calculate frequency of 0
        x = 1 - sum(data) / len(data)
        self._p = [(0, x), (x, 0.9999)]

    def encode(self, data):
        self._initialize(data)
        for i in data:
            l_i, h_i = self._p[i]
            d = self._h - self._l
            self._h = self._l + d * h_i
            self._l = self._l + d * l_i
            print "l:%f h:%f" % (self._l, self._h)
        r = {"payload": self._l, "model": self._p}
        return r

    def _dinitialize(self):
        self._l = 0
        self._h = 0.9999
        #calculate frequency of 0

    def decode(self, data):
        self._dinitialize()
        self._output = data["model"]
        n = data["payload"]
        while(n > 0):
            for i, (l_i, h_i) in it.izip(range(len(self._p)), self._p):
                if l_i <= n and n < h_i:
                    self._output.append(i)
                    d = h_i - l_i
                    n = (n - l_i) / d
                    break
        return self._output


class barithmeticb(object):

    """Class of a binary arithmetic codec implemented using integer
    arithmetic

    """

    _underflow_bits = 0
    _l = 0
    _h = 1
    _buff = 0
    _output = []
    _p = []
    _bit_size = 0
    _scale = 0

    def __init__(self, bit_size=16):
        super(arithmeticb, self).__init__()
        self._bit_size = bit_size
        self._scale = 2 ** self._bit_size - 1

    def _initialize(self, data):
        self._ctr = 0
        self._l = 0
        self._h = self._scale
        self._buff = 0
        self._output = []
        #calculate frequency of 0
        x = 1 - sum(data) / len(data) * self._scale
        self._p = [(0, x), (x, self._scale)]

    def encode(self, data):
        self._initialize(data)
        for i in data:
            l_i, h_i = self._p[i]
            d = self._h - self._l
            self._h = self._l + d / self._scale * h_i
            self._l = self._l + d / self._scale * l_i
            self._check_underflow()
            print "l:%f h:%f" % (self._l, self._h)
        r = {"payload": self._l, "model": self._p}
        return r

    def _check_overflow(self):
        MSB = 1 << (self._scale - 1)
        if self._h & MSB is 1 and self._l & MSB is 1:
            self._output.append(1)
            for i in range(self._underflow_bits):
                self._output.append(0)
                self._shift()
        else:
            MSB = 1 << (self._scale << 2)
            if self._h & MSB is 1 and self._l & MSB is 0:
                self._underflow_bits += 1
                self._l &=

    def _shift(self):
        self._l <<= 1
        self._h <<= 1
        self._h |= 1

    def _dinitialize(self):
        self._l = 0
        self._h = 0.9999
        #calculate frequency of 0

    def decode(self, data):
        self._dinitialize()
        self._output = data["model"]
        n = data["payload"]
        while(n > 0):
            for i, (l_i, h_i) in it.izip(range(len(self._p)), self._p):
                if l_i <= n and n < h_i:
                    self._output.append(i)
                    d = h_i - l_i
                    n = (n - l_i) / d
                    break
        return self._output
