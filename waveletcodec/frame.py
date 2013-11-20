"""Frame class definition.

.. :module: frame
    platform: Unix, Windows

.. :author: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""


class Frame:

    """Represent a video frame.

    This class represent a video frame that holds a payload and a header.
    The header is a dictionary that holds information about the payload such
    as frame number, encoding algorithm name, etc.

    """

    payload = None
    header = {}

    def __init__(self):
        pass

    def set_payload(self, data):
        """Set the frame payload.

        Args:
            data: data to be bound to the frame payload

        Return:
            None

        """
        self.payload = data
        return

    def set_hattr(self, name, value):
        """Set a new attribute for the frame.

        This method set a new attribute. Such attribute is set inside the
        header of the frame.

        Args:
            name (string): the name of the attribute
            value (multi): the value of the attribute

        Return:
            None

        """
        self.header[name] = value
        return

    def get_hattr(self, name):
        """Get the value of an attribute.

        Args:
            name (string): Name of the attribute

        Return:
            The value of the requested attribute

        Raises:
            IndexError

        """
        if name not in self.header:
            raise IndexError("The attribute %s is not set" % name)
        return self.header[name]

    def to_bitstream():
        #TODO
        """Encode the frame to a bit stream.

        This method creates a bitstream combining the header and the payload
        into one BLOB
        """
        pass

    @staticmethod
    def from_bitstream(bitstream):
        #TODO
        """Decode a frame from a bitstream

        This method create a new frame using the given bitstream by decoding
        the header and the payload from it.

        Args:
            bitstream (): The bitstream to be decoded

        Return:
            a Frame instance that holds the data of the bitstream decoded

        """
        pass
