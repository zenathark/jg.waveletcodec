"""Collection of video codecs.

.. module::sources
   :platform: Unix, Windows

.. modelauthor:: Juan C Galan-Hernandez <jcgalanh@gmail.com>

"""

import waveletcodec.abstractsource
import os
import pickle
import cv2
from waveletcodec.frame import Frame
import tools


class RawPNGSource(waveletcodec.abstractsource.AbstractVideoSource):

    """This class creates a video source from png files.

    This class takes a given directory and create or reproduce a secuence
    of png images as a video. In the same directory should be stored a header
    file. Currently, the header is stored using pickle.

    """

    _path = ""
    _mode = cv2.IMREAD_COLOR
    _frame = 0

    def __init__(self, path, mode=cv2.IMREAD_COLOR):
        if path[-1] is not "/":
            path.append("/")
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            try:
                hfile = open(path + "header.dat", "rb")
                self._header = pickle.load(hfile)
                hfile.close()
            except IOError:
                self.set_hattr('frames', 0)
        self._path = path
        self._mode = mode
        self._frame = 0

    def next(self):
        idx = self._frame
        new_frame = Frame()
        new_frame.set_hattr('name', str(idx))
        new_frame.set_payload(cv2.imread(self._path + str(idx) + ".png",
                                         self._mode))
        self._frame += 1
        return new_frame

    def save_header(self):
        try:
            hfile = open(self._path + "header.dat", "wb")
            pickle.dump(self._header, hfile)
        except IOError, e:
            raise e

    def append(self, img):
        tools.check_dim(img, 2)
        name = self.get_hattr('frames')
        cv2.imwrite(self._path + str(name) + "PNG")
        self.set_hattr('frames', name + 1)

    def seek(self, index):
        if index < 0 or index >= self.get_hattr('frames'):
            raise IndexError("Index out of bounds")
        self._frame = index

    def reset(self):
        self._frame = 0
