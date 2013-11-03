"""Test unit for the tools module.

.. module:: test_tools
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import unittest
import waveletcodec.tools as tls
import numpy as np


class TsetUnitTools(unittest.TestCase):

    """Unit test for the tools module."""

    def test_mse(self):
        """Test case for the mse method.

        Returns:
            Nothing

        """
        mtx = np.ones((3, 3))
        mse = tls.mse(mtx, mtx)
        self.assertEquals(mse, 0, "Error mse %d" % mse)


if __name__ == '__main__':
    unittest.main()
