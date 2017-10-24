#!/bin/env python

from __future__ import division
import cv2
import os
import pickle
import numpy as np
import waveletcodec.intraframe as intra
import waveletcodec.sortspeck as sk
import waveletcodec.tools as tls
import waveletcodec.h264 as h264
import waveletcodec.entropy as etr
import waveletcodec.wave as wave
import getopt
import sys

__qstep = 40
__imshape = [256.0, 256.0]

__base_path = "test_db"
# __original = "/still_original/512/"
__original = "/videoframe_original/256/"
__fullsearch = "/fullsearch/"
__speck_path = "/speck/"
__fvspeck_path = "/fvspeck/"
__h265 = "/h265/"


def compress_h265(img):
    abcdc = etr.abac([0, 1])
    cdc = h264.integDCT()
    iimg = cdc.dct4x4(img, __qstep)
    encoded = abcdc.encode(h264.binarizate(iimg))
    return (encoded, iimg)


def decompress_h265(iimg):
    cdc = h264.integDCT()
    img = cdc.idct4x4(iimg, __qstep, __imshape)
    return img


def compress_speck(img, bpp):
    codec = sk.ar_speck()
    wavelet = tls.quantize(wave.cdf97(img, 5),
                           1000,
                           dtype=int)
    coded_img = codec.compress(wavelet, bpp)
    return coded_img


def decompress_speck(qwave):
    wavelet = tls.quantize(qwave,
                           0.001)
    img = wave.icdf97(wavelet)
    return img


def compress_fvspeck(img, bpp):
    codec = sk.ar_fvspeck()
    wavelet = tls.quantize(wave.cdf97(img, 5),
                           1000,
                           dtype=int)
    coded_img = codec.compress(wavelet, bpp, 0.06, (128, 128), 0.3, 1, 0.5)
    return coded_img


def decompress_fvspeck(qwave):
    wavelet = tls.quantize(qwave,
                           0.001)
    img = wave.icdf97(wavelet)
    return img


def process_dir_h265(path_in, path_out):
    for fimg in os.listdir(path_in):
        print("Compressing " + fimg)
        img = cv2.imread(path_in + fimg, cv2.IMREAD_GRAYSCALE)
        packt = compress_h265(img)
        print("Decompressing " + fimg)
        restored_img = decompress_h265(packt[1])
        np.save(open(path_out + fimg + ".dat", "wb"), packt[1])
        cv2.imwrite(path_out + fimg,
                    restored_img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        info = {"bpp": len(packt[0]['payload'])/(256*256)}
        print(info["bpp"])
        pickle.dump(info, open(path_out + fimg + ".inf", "wb"))


def process_dir_speck(path_in, path_out, path_compare):
    for fimg in os.listdir(path_in):
        last_info = pickle.load(open(path_compare + fimg + ".inf", "rb"))
        print("Compressing " + fimg + " at " + str(last_info['bpp']))
        img = cv2.imread(path_in + fimg, cv2.IMREAD_GRAYSCALE)
        packt = compress_speck(img, last_info['bpp'])
        print("Decompressing " + fimg)
        restored_img = decompress_speck(packt['clone'])
        cv2.imwrite(path_out + fimg,
                    restored_img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        pickle.dump(packt, open(path_out + fimg + ".inf", "wb"))


def process_dir_fvspeck(path_in, path_out, path_compare):
    for fimg in os.listdir(path_in):
        last_info = pickle.load(open(path_compare + fimg + ".inf", "rb"))
        print("Compressing " + fimg + " at " + str(last_info['bpp']))
        img = cv2.imread(path_in + fimg, cv2.IMREAD_GRAYSCALE)
        packt = compress_fvspeck(img, last_info['bpp'])
        print("Decompressing " + fimg)
        restored_img = decompress_fvspeck(packt['clone'])
        cv2.imwrite(path_out + fimg,
                    restored_img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        pickle.dump(packt, open(path_out + fimg + ".inf", "wb"))


if __name__ == '__main__':
    process_dir_h265("{0}{1}".format(__base_path, __original),
                     "{0}{1}".format(__base_path, __h265))

    process_dir_speck("{0}{1}".format(__base_path, __original),
                      "{0}{1}".format(__base_path, __speck_path),
                      "{0}{1}".format(__base_path, __h265))

    process_dir_fvspeck("{0}{1}".format(__base_path, __original),
                        "{0}{1}".format(__base_path, __fvspeck_path),
                        "{0}{1}".format(__base_path, __h265))
