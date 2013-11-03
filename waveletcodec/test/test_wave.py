"""test_wave.py.

.. module:: test_wave
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import unittest
import waveletcodec.wave as wave
import numpy as np
import numpy.testing as npt
import cv2


class TestWave(unittest.TestCase):

    """Test unit of the wave module.

    This is the unit test of the wave module

    """

    def test_WCSet_creation(self):
        """Test case for creation of a WCSet.

        This test case checks for a correct creation of a WCSet from a
        numpy.ndarray instance

        """
        fakewave = np.eye(2 ** 2)
        obj = wave.WCSet(fakewave, 2, wave.CDF97)
        self.assertIsInstance(obj, wave.WCSet, "Obj class missmatch")
        npt.assert_array_equal(obj, fakewave)
        self.assertEqual(obj.level, 2, "Failed to read level")
        self.assertEqual(obj.filter, wave.CDF97, "Failed to read filter type")

    def test_cdf97(self):
        signal = np.ones((2 ** 6, 2 ** 6))
        wavelet = wave.cdf97(signal)
        isignal = wave.icdf97(wavelet)
        npt.assert_array_almost_equal(signal, isignal, 6)

    def test_to_image(self_):
        signal = cv2.imread('docs/lena512color.tiff', cv2.IMREAD_GRAYSCALE)
        wavelet = wave.cdf97(signal, 3)
        display = wavelet.as_image()
        self.assertEqual(display.dtype, np.uint8, "Not an image")

if __name__ == '__main__':
    unittest.main()
