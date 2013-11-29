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
