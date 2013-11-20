"""Test unit for Frame module

.. :module: test_frame
    platform: Unix, Windows

.. :author: Juan C. Galan-Henrnadez <jcgalanh@gmail.com>

"""

import unittest
from waveletcodec.frame import Frame


class TestFrame(unittest.TestCase):

    """Test unit for the frame module """

    def test_setattr(self):
        """Test of the seattr method """
        frame = Frame()
        fake_attr = 'Meh'
        frame.set_hattr('fake', fake_attr)
        frame.set_hattr('index', 0)
        self.assertEqual(fake_attr,
                         frame.get_hattr('fake'),
                         "New attribute not set")
        self.assertEqual(0,
                         frame.get_hattr('index'),
                         "index not set")
        with self.assertRaises(IndexError):
            frame.get_hattr('nonexistant')


if __name__ == '__main__':
    unittest.main()
