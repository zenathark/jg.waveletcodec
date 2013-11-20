"""Video codec definition class.

.. module::videocodec
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import waveletcodec.abstractcodec


class VideoCoder(object):

    """This class represents a simple video codec.

    A simple video codec takes both an abstract coder and decoder in order
    to create a video stream.

    """

    _coder = None
    _decoder = None
    _current_frame = 0
    _source = None

    def __init__(self, coder, decoder, source=None):
        if not isinstance(coder, waveletcodec.abstractcodec.Encoder):
            raise TypeError("Type invalid for the coder, must be a \
                            waveletcodec.abstractcodec.Encoder type")
        if not isinstance(decoder, waveletcodec.abstractcodec.Decoder):
            raise TypeError("Type invalid for the coder, must be a \
                            waveletcodec.abstractcodec.Decoder type")
        self._coder = coder
        self._decoder = decoder
        if source is not None:
            self.set_source(source)

    def set_source(self, source):
        """Set a new source of video.

        This method sets a new source of a video stream and resets the
        current frame position to 0

        Args:
            source (waveletcodec.AbstractSource): A video source

        Return:
            None

        """
        self._source = source
        self._current_frame = 0
