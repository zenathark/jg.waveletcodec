"""Basic codec contract.

.. module::abstractcodec
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import abc


class Encoder(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, image):
        """Encode a new image into the video stream.

        Args:
            image (numpy.ndarray): The data to be encoded into the video
            stream

        Returns:
            An instance of waveletcodec.Frame that contains the encoded frame
            and a header

        """
        return


class Decoder(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def decode(self, frame):
        """Decode a frame from the video stream

        Args:
            frame (waveletcodec.Frame): The frame to be decoded

        Returns:
            An instance of numpy.ndarray that holds the decoded image

        """
        return
