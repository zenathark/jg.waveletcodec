"""A simple video codec that only uses interframe coding.

.. module::motioncodec
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import abc
import abstractcodec


class MotionEncoder(abstractcodec.Encoder):

    """A simple video encoder.

    This class represents a simple video encoder that only compresses a single
    video frame using a spatial compressor.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _compress(self, image):
        """Compress an image.

        Args:
            image (numpy.ndarray): The image to be compressed

        Returns:
            An instance of waveletcodec.Frame that holds the compressed stream

        """
        return

    def encode(self, image):
        return self._compress(image)


class MotionDecoder(abstractcodec.Decoder):

    """A simple video decoder.

    This class represents a simple video decoder that decompresses a single
    video frame using a spatial compressor.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _decompress(self, frame):
        """Decode a compresed frame.

        Args:
            frame (waveletcodec.Frame): The frame to be expanded

        Return:
            An instance of numpy.ndarray that holds the reconstructed image

        """
        return

    def decode(self, frame):
        return self._decompress(frame)
