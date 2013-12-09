"""Test unit for entropy module

.. :module: test_entropy
    platform: Unix, Windows

.. :author: Juan C. Galan-Henrnadez <jcgalanh@gmail.com>

"""

import unittest
import waveletcodec.entropy


class TestFrame(unittest.TestCase):

    """Test unit for the frame module """

    def test_barithmeticb(self):
        """Test of the arithmetic coding class """
        # payload = list(np.random.random_integers(0, 1, 3))
        sigma = ['S', 'W', 'I', 'M', ' ']
        payload = list("SWISS MISS")
        codec = waveletcodec.entropy.barithmeticb(sigma, 8)
        stream = codec.encode(payload)
        print stream

if __name__ == '__main__':
    unittest.main()
