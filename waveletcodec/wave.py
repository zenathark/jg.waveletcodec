"""Represent a Wavelet Coefficient Set.

.. module::wave
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import numpy as np
import waveletcodec.tools as tools
import waveletcodec.lwt as lwt
import cv2
import math

#Constant Section
CDF97 = 1

#End


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
        print(cls)
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

    def as_image(self):
        dc_rows, dc_cols = self.shape
        dc_rows //= 2 ** self.level
        dc_cols //= 2 ** self.level
        dc = self.copy()
        ac = dc[:dc_rows, :dc_cols].copy()
        dc[:dc_rows, :dc_cols] = 0
        ac = tools.normalize(ac, upper_bound=255, dtype=np.uint8)
        dc = np.abs(dc)
        dc = tools.normalize(dc, upper_bound=255, dtype=np.uint8)
        #ac = cv2.equalizeHist(ac)
        dc = cv2.equalizeHist(dc)
        dc[:dc_rows, :dc_cols] = ac
        return dc


_CDF97 = lwt.FilterBank(
    scale=1 / 1.149604398,
    update=[-0.05298011854, 0.4435068522],
    predict=[-1.586134342, 0.8829110762]
)


def cdf97(signal, level=1):
    """Calculate the Wavelet Transform of the signal using the CDF97 wavelet.

    This method calculates the LWT of the signal given using the
    Cohen-Daubechies-Feauveau wavelet using a filter bank of size 9,7

    Args:
        signal a 1D or 2D numpy.array instance

    Returns:
        An instance of Wavelet that holds the coefficients of the transform

    """
    coeff = _CDF97.forward(signal, level)
    wavelet = WCSet(coeff, level, CDF97)
    return wavelet


def icdf97(wavelet):
    """Calculate the inverse Wavelet Transform using the CDF97 wavelet.

    This method calculates the iLWT of the wavelet given using the
    Cohen-Daubechies-Feauveau wavelet using a filter bank of size 9,7

    Args:
        wavelet a 1D or 2D Wavelet instance

    Returns:
        An instance of numpy.ndarray that holds the reconstructed signal

    """
    signal = _CDF97.inverse(wavelet, wavelet.level)
    return signal


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
        mtx += [(y, x)]
    return mtx


# def get_morton_order(dim, idx = 0, size = -1):
#     if size < 0:
#         mtx = deque()
#     else:
#         mtx = deque([],size)
#     if idx <> 0:
#         swp = idx
#         idx = dim
#         dim = swp
#     n = int(math.log(dim,2))
#     pows = range(int(n/2))
#     for i in range(dim):
#         x = 0
#         y = 0
#         for j in pows:
#             x |= ((i >> 2*j) & 1) << j
#             y |= ((i >> 2*j+1) & 1) << j
#         if idx == 0:
#             mtx += [vector((y,x))]
#         else:
#             idx -= 1
#     return mtx
