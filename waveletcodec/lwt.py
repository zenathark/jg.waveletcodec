"""Lifting Wavelet Transform Module.

File: lwt.py
Author: jcgalanh@gmail.com
Description: This module contains an implementation of the lifting wavelet
transform for common wavelets used in compression such as the cdf9 and the
cdf 53. The lwt is only implemented for one and two dimensional signals.

"""
import numpy as np
import waveletcodec.wave as wv
import waveletcodec.tools as tl

CDF97 = 1
CDF53 = 2


class FilterBank(object):

    """
    A wavelet filter bank representation.

    This object represents a filter bank used on the lifting wavelet
    transform. The filter bank must have a scale coefficient and an array
    of update and predict coefficients. The update and predict coefficients
    must have the same length

    """

    def __init__(self, scale, update, predict):
        super(FilterBank, self).__init__()
        if update.__class__ is not list:
            raise TypeError("update must be a list")
        if predict.__class__ is not list:
            raise TypeError("predict must be a list")
        if len(predict) is not len(update):
            raise IndexError("predict and update must be of the same size")
        self._scale = scale
        self._update = update
        self._predict = predict
        self._length = len(update)

    def _forward(self, signal):
        """Calculate a one level forward lifting wavelet transform (LWT).

        Calculate a forward wavelet transform using the lifting scheme. The
        filter bank used is the one specified at instantiation time. This
        method only calculates a one level decomposition.

        Args:
            signal The signal to be processed using the LWT. The signal must be
            a one dimentional matrix

        Returns:
            A new Wavelet object holding the result of the transform, the
            default data type is numpy.float64

        """
        for i in np.arange(0, self._length):
            #predict
            signal[1:-1:2] += self._predict[i] * (signal[:-2:2] + signal[2::2])
            signal[-1] += 2 * self._predict[i] * signal[-2]
            #update
            signal[2::2] += self._update[i] * (signal[1:-1:2] + signal[3::2])
            signal[0] += 2 * self._update[i] * signal[1]
        #scale
        signal[:2] *= self._scale
        signal[1::2] /= self._scale
        #sort lazy wavelet
        signal = _sort(signal)
        return signal

    def _inverse(self, wavelet):
        """Calculate the inverse fast wavelet transform (iLWT).

        Calculate the iLWT of a one dimentional signal.

        Args:
            wavelet the wavelet to be reversed to a one dimentional signal

        Returns:
            A one dimentional instance of a numpy.array

        """

        #undo scale
        wavelet = _unsort(wavelet)
        wavelet[:2] *= (1 / self._scale)
        wavelet[1::2] /= (1 / self._scale)
        for i in np.arange(0, self._length):
            #Undo update
            wavelet[2::2] += self._update[self._length - i - 1] * \
                (wavelet[1:-1:2] + wavelet[3::2])
            wavelet[0] += 2 * self._update[i] * wavelet[1]
            #Undo predict
            wavelet[1:-1:2] += self._predict[self._length - i - 1] * \
                (wavelet[:-2:2] + wavelet[2::2])
            wavelet[-1] += 2 * self._predict[i] * wavelet[-2]
        return wavelet

    def _forward2D(self, signal):
        """Calculate a one level forward lifting wavelet transform (LWT).

        Calculate a forward wavelet transform using the lifting scheme. The
        filter bank used is the one specified at instantiation time. This
        method only calculates a one level decomposition.

        Args:
            signal The signal to be processed using the LWT. The signal must be
            a two dimentional matrix

        Returns:
            A new Wavelet object holding the result of the transform, the
            default data type is numpy.float64

        """
        for i in np.arange(0, self._length):
            #predict
            signal[:, 1:-1:2] += self._predict[i] * \
                (signal[:, :-2:2] + signal[:, 2::2])
            signal[:, -1] += 2 * self._predict[i] * signal[:, -2]
            #update
            signal[:, 2::2] += self._update[i] * \
                (signal[:, 1:-1:2] + signal[:, 3::2])
            signal[:, 0] += 2 * self._update[i] * signal[:, 1]
        #scale
        signal[:, :2] *= self._scale
        signal[:, 1::2] /= self._scale
        #sort lazy wavelet
        signal = _sort2D(signal)
        return signal

    def _inverse2D(self, wavelet):
        """Calculate the 2D inverse fast wavelet transform (iLWT).

        Calculate the iLWT of a two dimentional signal.

        Args:
            wavelet the wavelet to be reversed to a two dimentional signal

        Returns:
            A two dimentional instance of a numpy.array

        """

        #undo scale
        wavelet = _unsort2D(wavelet)
        wavelet[:, :2] *= (1 / self._scale)
        wavelet[:, 1::2] /= (1 / self._scale)
        for i in np.arange(0, self._length):
            #Undo update
            wavelet[:, 2::2] += self._update[self._length - i - 1] * \
                (wavelet[:, 1:-1:2] + wavelet[:, 3::2])
            wavelet[:, 0] += 2 * self._update[i] * wavelet[:, 1]
            #Undo predict
            wavelet[:, 1:-1:2] += self._predict[self._length - i - 1] * \
                (wavelet[:, :-2:2] + wavelet[:, 2::2])
            wavelet[:, -1] += 2 * self._predict[i] * wavelet[:, -2]
        return wavelet


def _sort(signal):
    """Sort an in-place LWT.

    This method sorts an in place LWT of a signal. After an in place LWT
    the odd coefficients must be placed on the first half of the wavelet
    and the event coefficients must be placed ont the last half

    Args:
        signal a one dimentional numpy.ndarray matrix

    Returns:
        A new instance of numpy.ndarray with the coefficients sorted out

    """

    to = signal.shape[0]
    for i in range(1, to / 2 + 1, 1):
        temp = signal[i].copy()
        signal[i:to - 1] = signal[i + 1:to]
        signal[-1] = temp
    return signal


def _unsort(signal):
    """Revert the operation of _sort.

    Args:
        signal an instance of numpy.ndarray of one dimention

    Returns:
        An instance of numpy.ndarray

    """
    to = signal.shape[0]
    for i in range(to / 2, 0, -1):
        temp = signal[-1].copy()
        signal[i + 1:] = signal[i:-1]
        signal[i] = temp
    return signal


def _sort2D(signal):
    """Revert the operation of _sort.

    Args:
        signal an instance of numpy.ndarray of one dimention

    Returns:
        An instance of numpy.ndarray

    """
    to = signal.shape[1]
    for i in range(1, to / 2 + 1, 1):
        temp = signal[:, i].copy()
        signal[:, i:to - 1] = signal[:, i + 1:to]
        signal[:, -1] = temp
    return signal


def _unsort2D(signal):
    """Sort an in-place LWT.

    This method sorts an in place LWT of a signal. After an in place LWT
    the odd coefficients must be placed on the first half of the wavelet
    and the event coefficients must be placed ont the last half

    Args:
        signal a two dimentional numpy.ndarray matrix

    Returns:
        A new instance of numpy.ndarray with the coefficients sorted out

    """
    to = signal.shape[1]
    for i in range(to / 2, 0, -1):
        temp = signal[:, -1].copy()
        signal[:, i + 1:] = signal[:, i:-1]
        signal[:, i] = temp
    return signal

_CDF97 = FilterBank(
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

    return _CDF97._forward2D(signal)


def icdf97(wavelet):
    """Calculate the inverse Wavelet Transform using the CDF97 wavelet.

    This method calculates the iLWT of the wavelet given using the
    Cohen-Daubechies-Feauveau wavelet using a filter bank of size 9,7

    Args:
        wavelet a 1D or 2D Wavelet instance

    Returns:
        An instance of numpy.ndarray that holds the reconstructed signal

    """

    return _CDF97._inverse(wavelet)
# def cdf97(signal, level=1):
#     tl.check_ndarray(signal)
#     signal = normal_forward(signal,level,1/1.149604398,(-1.586134342,-0.05298011854,0.8829110762,0.4435068522))
#     return wv.wavelet2D(signal,level,"cdf97")
#
# def cdf53(signal, level = 1, in_place = True):
#     if not isinstance(signal, np.ndarray) or signal.ndim != 2:
#         raise TypeError, "Signal expected as 2D ndarray (numpy)"
#     if not in_place:
#         signal = signal.copy()
#     if signal.dtype == np.uint8:
#         sig_i8 = np.zeros((len(signal),2*len(signal[0])),np.uint8)
#         sig_i8[:,0:-1:2] = signal
#         sig_i16 = sig_i8.view(np.uint16)
#         sig_f16 = sig_i16.view(np.float16)
#         sig_f16[:] = sig_i16
#         signal = sig_f16
#     signal = normal_forward(signal,level,math.sqrt(2),(-0.5,0.25))
#     return wv.wavelet2D(signal,level,"cdf53")
#
# def icdf97(wave, in_place = True):
#     if not isinstance(wave, wv.wavelet2D):
#         raise TypeError, "Signal expected as wavelet2D"
#     signal = wave.data.copy()
#     #signal.dtype = np.float32
#     #signal[:] = wave.data
#     signal = normal_inverse(signal,wave.level,1.149604398,(-0.4435068522,-0.8829110762,0.05298011854,1.586134342))
#     return signal
#
# def icdf53(wave, in_place = True):
#     if not isinstance(wave, wv.wavelet2D):
#         raise TypeError, "Signal expected as wavelet2D"
#     signal = wave.data.copy()
#     signal.dtype = np.float32
#     signal[:] = wave.data
#     signal = normal_inverse(signal,wave.level,1/math.sqrt(2),(-0.25,0.5))
#     return signal

# def normal_forward(signal, level,  scale_coeff, coeff):
#     decomposed_signal = signal
#     for x in range(level):
#         forward(decomposed_signal, scale_coeff, coeff)
#         decomposed_signal = forward(decomposed_signal.T, scale_coeff, coeff).T
#         updated_rows = int(len(decomposed_signal) / 2)
#         updated_cols = int(len(decomposed_signal[0]) / 2)
#         decomposed_signal = decomposed_signal[:updated_rows,:updated_cols]
#     return signal
#
# def normal_inverse(signal, level, scale_coeff, coeff):
#     updated_rows = len(signal) / 2 **(level-1)
#     updated_cols = len(signal[0]) / 2 **(level-1)
#     for x in range(level):
#         recomposed_signal = signal[:updated_rows,:updated_cols]
#         recomposed_signal = inverse(recomposed_signal.T, scale_coeff, coeff).T
#         recomposed_signal = inverse(recomposed_signal, scale_coeff, coeff)
#         updated_rows = updated_rows * 2
#         updated_cols = updated_cols * 2
#     return signal
#
#
# def inverse(signal, scale_coeff, coeff):
#     if not isinstance(signal, np.ndarray) or signal.ndim != 2:
#         raise TypeError, "Signal expected as 2D ndarray (numpy)"
#     if len(coeff) <= 0:
#         raise TypeError, "Filter bank empty"
#     if len(coeff) % 2 != 0:
#         raise TypeError, "Filter bank expected to be Predict-Update pairs"
#     #undo scale
#     signal = unsort(signal)
#     signal[:,:2] *= scale_coeff
#     signal[:,1::2] /= scale_coeff
#     for i in np.arange(0,len(coeff),2):
#         #Undo update
#         signal[:,2::2] += coeff[i] * (signal[:,1:-1:2] + signal[:,3::2])
#         signal[:,0] += 2 * coeff[i] * signal[:,1]
#         #Undo predict
#         signal[:,1:-1:2] += coeff[i+1] * (signal[:,:-2:2] + signal[:,2::2])
#         signal[:,-1] += 2 * coeff[i+1] * signal[:,-2]
#     return signal
