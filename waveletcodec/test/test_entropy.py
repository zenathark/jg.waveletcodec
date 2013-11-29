"""Test unit for entropy module

.. :module: test_entropy
    platform: Unix, Windows

.. :author: Juan C. Galan-Henrnadez <jcgalanh@gmail.com>

"""

import unittest
import waveletcodec.entropy
import numpy as np


class TestFrame(unittest.TestCase):

    """Test unit for the frame module """

    def test_barithmeticb(self):
        """Test of the arithmetic coding class """
        codec = waveletcodec.entropy.barithmeticb()
        # payload = list(np.random.random_integers(0, 1, 3))
        payload = [1, 0, 1, 0]
        stream = codec.encode(payload)
        # istream = codec.decode(stream)
        # self.assertEqual(payload,
        #                  istream,
        #                  "Encoding error")

if __name__ == '__main__':
    unittest.main()
