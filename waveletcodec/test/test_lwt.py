"""Unit test for the lwt module.

File: test_lwt.py
Author: jcgalanh@gmail.com
Description: Test Unit for the lwt module

"""

import unittest
import waveletcodec.lwt as lwt
import numpy as np
import numpy.testing as npt

class TestLWT(unittest.TestCase):

    def test_cdf97(self):
        signal = np.ones((2**2,2**2))
        wavelet = lwt.cdf97(signal)
        isignal = lwt.icdf97(wavelet)
        npt.assert_array_equal(signal, isignal)

if __name__ == '__main__':
    unittest.main()
