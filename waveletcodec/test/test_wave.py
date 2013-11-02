"""test_wave.py.

.. module:: test_wave
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import unittest
import waveletcodec.wave as wave
import numpy as np
import numpy.testing as npt


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


if __name__ == '__main__':
    unittest.main()
