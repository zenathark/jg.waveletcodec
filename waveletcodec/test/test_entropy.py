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
        sigma = ['S', 'W', 'I', 'M']
        payload = list("SWISSMISS")
        codec = waveletcodec.entropy.barithmeticb(sigma)
        stream = codec.encode(payload)
        py = stream['payload']
        start = [str(i) for i in py[:16]]
        print stream
        print int(''.join(start), 2)
        # istream = codec.decode(stream)
        # self.assertEqual(payload,
        #                  istream,
        #                  "Encoding error")

if __name__ == '__main__':
    unittest.main()
