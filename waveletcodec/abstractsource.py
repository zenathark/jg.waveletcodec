"""A base class for a video source.

.. module::abstractsource
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import abc


class AbstractVideoSource(object):
    __metaclass__ = abc.ABCMeta

    _header = {}

    @abc.abstractmethod
    def get_frame(self, index):
        """Return a frame at a given position.

        Args:
            index (int): the index of the wanted frame

        Returns:
            An instance of waveletcodec.Frame that holds the requested frame
        """
        return

    def set_hattr(self, name, value):
        """Sets a new value to the source header.

        Args:
            name (string): The name of the new or existing attribute
            value (mixed): The new value of the attribute

        Returns:
            None

        """
        self._header[name] = value

    def get_hattr(self, name):
        """Get a value from the header.

        Gets a value from an attribute of the header. The attribute must
        exist.

        Args:
            name (string): The name of the attribute

        Return
            The value of the attribute (mixed)

        """
        if name not in self._header:
            raise AttributeError("The attribute '%s' does not exists" % name)
        return self._header[name]
