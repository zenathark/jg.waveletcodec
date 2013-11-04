"""Unit test for the lwt module.

.. module:: test_lwt
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import unittest
import waveletcodec.lwt as lwt
import numpy as np
import numpy.testing as npt
import waveletcodec.wave as wave


class TestLWT(unittest.TestCase):

    """Test unit for the LWT module.

    This is the test unit for the Lifting Wavelet Transform module

    """

    def test_forward_inverse(self):
        signal = np.arange(2 ** 4)
        wavelet = wave._CDF97._forward(signal)
        isignal = wave._CDF97._inverse(wavelet)
        npt.assert_array_almost_equal(signal, isignal, 6,
                                      "inverse wavelet failed")

    def test_forward_inverse2D(self):
        signal = np.ones((2 ** 4, 2 ** 4))
        wavelet = wave._CDF97._forward2D(signal)
        isignal = wave._CDF97._inverse2D(wavelet)
        npt.assert_array_almost_equal(signal, isignal, 6,
                                      "inverse wavelet failed")

    def test_forward_inverse_n(self):
        signal = np.arange(2 ** 8)
        wavelet = wave._CDF97._forward_n(signal, 2)
        isignal = wave._CDF97._inverse(wavelet)
        npt.assert_array_almost_equal(signal, isignal, 6,
                                      "inverse wavelet failed")

    def test_forward_inverse2D_n(self):
        signal = np.ones((2 ** 8, 2 ** 8))
        wavelet = wave._CDF97._forward2D_n(signal, 3)
        isignal = wave._CDF97._inverse2D_n(wavelet, 3)
        npt.assert_array_almost_equal(signal, isignal, 6,
                                      "inverse wavelet failed")


if __name__ == '__main__':
    unittest.main()
