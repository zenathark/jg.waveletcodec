"""Unit test for the lwt module.

.. module:: test_lwt
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import unittest
import waveletcodec.lwt as lwt
import numpy as np
import numpy.testing as npt


class TestLWT(unittest.TestCase):

    """Test unit for the LWT module.

    This is the test unit for the Lifting Wavelet Transform module

    """

    def test_forward_inverse(self):
        signal = np.arange(2 ** 4)
        wavelet = lwt._CDF97._forward(signal)
        isignal = lwt._CDF97._inverse(wavelet)
        npt.assert_allclose

    def test_cdf97(self):
        signal = np.ones((2**2,2**2))
        wavelet = lwt.cdf97(signal)
        isignal = lwt.icdf97(wavelet)
        npt.assert_array_equal(signal, isignal)

if __name__ == '__main__':
    unittest.main()
