"""Module for Entropy Coding Algorithms.

.. module::entropy
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

from __future__ import division
import itertools as it
import collections as cl


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
    _bit_size = 0
    _scale = 0
    _sigma = []
    _frequency = {}
    _model = {}

    def __init__(self, sigma, bit_size=16, **kargs):
        super(barithmeticb, self).__init__()
        self._bit_size = bit_size
        self._sigma = sigma
        self._scale = 2 ** self._bit_size - 1
        if 'model' in kargs:
            self._model = kargs['model']
        else:
            self._model = None

    def _initialize(self, data):
        self._l = 0
        self._h = self._scale
        self._buff = 0
        self._output = []
        #calculate frequency of 0
        if self._model is None:
            self._calculate_model(data)

    def encode(self, data):
        """ given list using arithmetic encoding."""
        self._initialize(data)
        for i in data:
            l_i, h_i = self._model[i]
            d = self._h - self._l
            self._h = int(self._l + d / self._scale * h_i)
            self._l = int(self._l + d / self._scale * l_i)
            self._check_overflow()
            print "l:%d h:%d" % (self._l, self._h)
        self._output = [int(i) for i in bin(self._l)[2:]] + self._output
        r = {"payload": self._output, "model": self._model}
        return r

    def _calculate_model(self, data):
        self._model = cl.OrderedDict()
        accum = 0
        for i in self._sigma:
            self._frequency[i] = data.count(i)
            p = (self._frequency[i] / len(data))
            p *= 2 ** self._bit_size - 1
            p = int(p)
            self._model[i] = (accum, accum + p)
            accum += p

    def _check_overflow(self):
        MSB = 1 << (self._bit_size - 1)
        if self._h & MSB == self._l & MSB:
            self._output.insert(0, int((self._h & MSB) > 0))
            for _ in range(self._underflow_bits):
                self._output.insert(0, ~(self._h & MSB) &
                                    (2 ** self._bit_size - 1) >>
                                    (2 ** self._bit_size - 1))
            self._underflow_bits = 0
            self._shift()
        else:
            self._check_underflow()

    def _check_underflow(self):
        MSB = 1 << (self._bit_size - 2)
        if self._h & MSB > 1 and self._l & MSB == 0:
            self._underflow_bits += 1
            low_mask = ((1 << self._bit_size - 1) |
                        (1 << self._bit_size - 2))
            low_mask = ~low_mask & 2 ** self._bit_size - 1
            self._l &= low_mask
            self._h |= (1 << self._bit_size - 1)
            self._shift()

    def _shift(self):
        self._l <<= 1
        self._h <<= 1
        self._l &= 2 ** self._bit_size - 1
        self._h &= 2 ** self._bit_size - 1
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
