"""Wavelet definition.

.. module:: wave
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import numpy as np
import waveletcodec.tools as tools
from waveletcodec.lwt import CDF97
from waveletcodec.lwt import icdf97


class WCSet(np.ndarray):

    """
    This object represents a wavelet.

    The fundamental element for signal processing using wavelets is an N matrix
    that holds the coefficients of a wavelet decomposition. This object extends
    from numpy.ndarray and extends it to hold the extra values needed for a
    wavelet data set

    """

    level = 0
    filter = None

    def __new__(cls, array, level, filter_=None):
        """Create a wavelet.

        This method creates a wavelet object using a numpy.ndarray as base

        Args:
            array. A numpy.ndarray as a base for this wavelet
            level. Level of decomposition of this wavelet
            filter. Filter bank name used

        Return:
            A Wavelet object with the same data as the numpy.ndarray object.
            The data is shared between both objects

        """
        print cls
        obj = np.asarray(array).view(cls)
        obj.level = level
        obj.filter = filter_
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.level = getattr(obj, 'level', None)
        self.filter = getattr(obj, 'filter', None)

    def inverse(self):
        """Return the inverse of this wavelet coefficients.

        This method returns the inverse transform of this wavelet
        as another numpy.ndarray matrix. The method chooses the apropiate
        inverse transform filter using the class property filter.

        Return:
            An numpy.ndarray instance that holds the reconstructed signal
            using the filter specified in the class property filter.

        Raises:
            AttributeError if the property filter is not set

        """
        if self.filter is None:
            msg = "filter property is not set, unable to determine the inverse"
            raise AttributeError(msg)
        if self.filter is CDF97:
            return icdf97(self)
