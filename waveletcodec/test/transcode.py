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


class MainHeader(object):
    frames = 0
    ext = ".png"


class IntraHeader(object):
    frames = 0
    ext = ".npy"
    search_size = 100
    block_size = 8
    motionext = ".mvn"


class HEVCHeader(object):
    frames = 0
    qstep = 40
    ext = ".npy"
    shape = 0


class HEVCFrameHeader(object):
    abac_size = 0


class SPECKFrameHeader(object):
    bpp = 0
    level = 0
    wavelet = ''
    shape = 0


def split_raw(filename, dest_file):
    '''This function transcode a file to a Wavelet Compressed Video Format
    '''
    if dest_file[-1] != "/":
        dest_file += "/"
    original = cv2.VideoCapture(filename)
    loaded, frame = original.read()
    if not os.path.exists(dest_file):
        os.makedirs(dest_file)
    total_frames = original.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    current_frame = original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    frame_count = 0
    info = MainHeader()
    while loaded and current_frame < total_frames:
        target_file = dest_file + str(int(frame_count)) + ".png"
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
        cv2.imwrite(target_file, frame, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0])
        loaded, frame = original.read()
        current_frame = original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        frame_count += 1
    info.frames = int(frame_count)
    info_file = dest_file + "header.dat"
    pickle.dump(info, open(info_file, "w"))


def compress_fullsearch(path, dest_path):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat", "r"))
    header = IntraHeader()
    header.search_size = 100
    header.block_size = 8
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    key_frame = cv2.imread(path + "0.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    if not cv2.imwrite(dest_path + "key.png",
                       key_frame, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
        print "Failed to create: key frame"
    for c in range(1, info.frames):
        frame = cv2.imread(path + str(c) + info.ext, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        error, mvs = intra.encode_motion_frame(frame, key_frame,
                                               8,
                                               100)
        np.save(open(dest_path + str(c) + header.ext, "w"), error)
        np.save(open(dest_path + str(c) + header.motionext, "w"), mvs)
    header.frames = info.frames
    header.ext = ".npy"
    pickle.dump(header, open(dest_path + "header.dat", "w"))

def compress_key_h265(path, dest_path):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat", "r"))
    header = HEVCHeader()
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    abcdc = etr.abac([0, 1])
    for c in range(info.frames):
        cdc = h264.integDCT()
        frame = cv2.imread(path + str(c) + info.ext, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        iframe = cdc.dct4x4(frame, header.qstep)
        header.shape = frame.shape
        abacf = abcdc.encode(h264.binarizate(iframe))
        fheader = HEVCFrameHeader()
        fheader.abac_size = len(abacf['payload'])
        np.save(open(dest_path + str(c) + header.ext, "w"), iframe)
        pickle.dump(fheader, open(dest_path + str(c) +".hdr", "w"))
    header.frames = info.frames
    pickle.dump(header, open(dest_path + "header.dat", "w"))


def decompress_key_h265(path, dest_path):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat", "r"))
    header = MainHeader()
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for c in range(info.frames):
        cdc = h264.integDCT()
        iframe = np.load(path + str(c) + info.ext)
        frame = cdc.idct4x4(iframe, info.qstep, info.shape)
        target_file = dest_path + str((c)) + ".png"
        cv2.imwrite(target_file, frame, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0])
    pickle.dump(header, open(dest_path + "header.dat", "w"))


def compress_error_h265(path, dest_path):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat", "r"))
    header = HEVCHeader()
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    abcdc = etr.abac([0, 1])
    for c in range(1, info.frames):
        cdc = h264.integDCT()
        frame = np.load(open(path + str(c) + info.ext))
        iframe = cdc.dct4x4(frame, header.qstep)
        header.shape = frame.shape
        abacf = abcdc.encode(h264.binarizate(iframe))
        fheader = HEVCFrameHeader()
        fheader.abac_size = len(abacf['payload'])
        np.save(open(dest_path + str(c) + header.ext, "w"), iframe)
        pickle.dump(fheader, open(dest_path + str(c) +".hdr", "w"))
    header.frames = info.frames
    pickle.dump(header, open(dest_path + "header.dat", "w"))


def decompress_error_h265(path, dest_path):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat", "r"))
    header = MainHeader()
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for c in range(1, info.frames):
        cdc = h264.integDCT()
        iframe = np.load(path + str(c) + info.ext)
        frame = cdc.idct4x4(iframe, info.qstep, info.shape)
        target_file = dest_path + str((c)) + ".npy"
        np.save(open(target_file, "w"), frame)
    pickle.dump(header, open(dest_path + "header.dat", "w"))


def decompress_fullsearch(path, dest_path, macroblock_size=8):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat", "r"))
    header = MainHeader()
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    key_frame = cv2.imread(path + "key.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    for c in range(1, info.frames):
        error = np.load(open(path + str(c) + info.ext))
        mvs = np.load(open(path + str(c) + info.motionext))
        frame = intra.decode_motion_frame(error,
                                          mvs,
                                          info.block_size,
                                          key_frame)
        if not cv2.imwrite(dest_path + str(c) + header.ext,
                           frame, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
            print "Failed to create: " + dest_path + str(c) + header.ext
    header.frames = info.frames
    pickle.dump(header, open(dest_path + "header.dat", "w"))


def compress_key_speck(path, dest_path, data_path, dec_path, dec_level):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat"))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not os.path.exists(dec_path):
        os.makedirs(dec_path)
    header = MainHeader()
    header.frames = info.frames
    header.ext = '.dat'
    codec = sk.ar_speck()
    for c in range(info.frames):
        frame = cv2.imread(path + str(c) + info.ext,
                           cv2.CV_LOAD_IMAGE_GRAYSCALE)
        h264data = pickle.load(open(data_path + str(c) + ".hdr"))
        rows = frame.shape[0]
        cols = frame.shape[1]
        frame = tls.zero_padding(frame)
        wavelet = wave.cdf97(frame, dec_level)
        wavelet = tls.quantize(wavelet, 1000, dtype=int)
        bpp = h264data.abac_size / (rows * cols)
        print bpp
        coded_frame = codec.compress(wavelet, bpp)
        wvlt = codec.clone
        k1 = tls.quantize(wvlt, 0.001)
        iframe = wave.icdf97(k1)
        coded_frame['real_cols'] = cols
        coded_frame['real_rows'] = rows
        iframe = tls.unpadding(iframe, (rows, cols))
        iframe2 = tls.normalize(iframe, upper_bound=255, dtype=np.uint8)
        if not cv2.imwrite(dec_path + str(c) + ".png",
                           iframe2, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
            print "Failed to create: " + dest_path + str(c) + ".png"
        pickle.dump(iframe, open(dec_path +str(c) + ".npy","w"))
        try:
            pickle.dump(coded_frame,
                        open(dest_path + str(c) + header.ext, "wb"))
        except:
            print "Failed to create: " + dest_path + str(c) + header.ext
    pickle.dump(header, open(dest_path + "header.dat", "w"))
    pickle.dump(header, open(dec_path + "header.dat", "w"))


# def decompress_key_speck(path, dest_path, data_path, dec_level):
#     if path[-1] != "/":
#         path += "/"
#     info = pickle.load(open(path + "header.dat"))
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#     header = MainHeader()
#     header.frames = info.frames
#     header.ext = '.png'
#     codec = sk.ar_speck()
#     for c in range(info.frames):
#         r = pickle.load(open(path + str(c) + info.ext))
#         wavelet = codec.expand(r['payload'], r['colums'], r['rows'],
#                                r['level'], r['wisebit'], wave.CDF97)
#         frame = tls.quantize(frame, 0.001)
#         frame = wave.icdf97(wavelet)
#         frame = tls.unpadding(frame, (frame['real_rows'],
#                                       frame['real_cols']))
#         if not cv2.imwrite(dest_path + str(c) + ".png",
#                            frame, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
#             print "Failed to create: " + dest_path + str(c) + ".png"
#     pickle.dump(header, open(dest_path + "header.dat", "w"))


def compress_error_speck(path, dest_path, data_path, dec_path, dec_level):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat"))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not os.path.exists(dec_path):
        os.makedirs(dec_path)
    header = MainHeader()
    header.frames = info.frames
    header.ext = '.dat'
    codec = sk.ar_speck()
    for c in range(1, info.frames):
        frame = np.load(path + str(c) + ".npy")
        h264data = pickle.load(open(data_path + str(c) + ".hdr"))
        rows = frame.shape[0]
        cols = frame.shape[1]
        frame = tls.zero_padding(frame)
        wavelet = wave.cdf97(frame, dec_level)
        wavelet = tls.quantize(wavelet, 1000, dtype=int)
        bpp = h264data.abac_size / (rows * cols)
        print bpp
        coded_frame = codec.compress(wavelet, bpp)
        wvlt = codec.clone
        k1 = tls.quantize(wvlt, 0.001)
        iframe = wave.icdf97(k1)
        coded_frame['real_cols'] = cols
        coded_frame['real_rows'] = rows
        iframe = tls.unpadding(iframe, (rows, cols))
        try:
            pickle.dump(coded_frame,
                        open(dest_path + str(c) + header.ext, "wb"))
            pickle.dump(iframe,
                        open(dec_path + str(c) + ".npy", "wb"))
        except:
            print "Failed to create: " + dest_path + str(c) + header.ext
    pickle.dump(header, open(dest_path + "header.dat", "w"))
    pickle.dump(header, open(dec_path + "header.dat", "w"))


# def decompress_error_speck(path, dest_path, data_path, dec_level):
#     if path[-1] != "/":
#         path += "/"
#     info = pickle.load(open(path + "header.dat"))
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#     header = MainHeader()
#     header.frames = info.frames
#     header.ext = '.png'
#     codec = sk.ar_speck()
#     for c in range(1, info.frames):
#         r = pickle.load(open(path + str(c) + info.ext))
#         wavelet = codec.expand(r['payload'], r['colums'], r['rows'], r['level'],
#                                r['wisebit'], wave.CDF97)
#         frame = wave.icdf97(wavelet, dec_level)
#         frame = tls.quantize(frame, 0.001, dtype=int)
#         frame = tls.unpadding(frame, (frame['real_rows'],
#                                       frame['real_cols']))
#         if not cv2.imwrite(dest_path + str(c) + ".png",
#                            frame, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
#             print "Failed to create: " + dest_path + str(c) + ".png"
#     pickle.dump(header, open(dest_path + "header.dat", "w"))


def compress_key_fvspeck(path, dest_path, data_path, dec_path, dec_level):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat"))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not os.path.exists(dec_path):
        os.makedirs(dec_path)
    header = MainHeader()
    header.frames = info.frames
    header.ext = '.dat'
    codec = sk.ar_fvspeck()
    for c in range(info.frames):
        frame = cv2.imread(path + str(c) + info.ext,
                           cv2.CV_LOAD_IMAGE_GRAYSCALE)
        h264data = pickle.load(open(data_path + str(c) + ".hdr"))
        rows = frame.shape[0]
        cols = frame.shape[1]
        frame = tls.zero_padding(frame)
        wavelet = wave.cdf97(frame, dec_level)
        wavelet = tls.quantize(wavelet, 1000, dtype=int)
        bpp = h264data.abac_size / (rows * cols)
        print bpp
        center = (int(rows / 2), int(cols / 2))
        coded_frame = codec.compress(wavelet, bpp, 0.006, center, 0.3, 1, 1)
        coded_frame['real_cols'] = cols
        coded_frame['real_rows'] = rows
        wvlt = codec.clone
        iframe = wave.icdf97(wvlt)
        iframe = tls.quantize(iframe, 0.001)
        iframe = tls.unpadding(frame, (rows, cols))
        iframe2 = tls.normalize(iframe, upper_bound=255, dtype=np.uint8)
        if not cv2.imwrite(dec_path + str(c) + ".png",
                           iframe2, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
            print "Failed to create: " + dest_path + str(c) + ".png"
        pickle.dump(iframe, open(dest_path +str(c) + ".npy","w"))
        try:
            pickle.dump(coded_frame,
                        open(dest_path + str(c) + header.ext, "wb"))
        except:
            print "Failed to create: " + dest_path + str(c) + header.ext
    pickle.dump(header, open(dest_path + "header.dat", "w"))
    pickle.dump(header, open(dec_path + "header.dat", "w"))


# def decompress_key_fvspeck(path, dest_path, data_path, dec_level):
#     if path[-1] != "/":
#         path += "/"
#     info = pickle.load(open(path + "header.dat"))
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#     header = MainHeader()
#     header.frames = info.frames
#     header.ext = '.png'
#     codec = sk.ar_speck()
#     for c in range(info.frames):
#         r = pickle.load(open(path + str(c) + info.ext))
#         wavelet = codec.expand(r['payload'], r['colums'], r['rows'], r['level'],
#                                r['wisebit'], r['Lbpp'], r['lbpp'], ['center'],
#                                r['alpha'], r['c'], r['gamma'])
#         frame = wave.icdf97(wavelet, dec_level)
#         frame = tls.quantize(frame, 0.001, dtype=int)
#         frame = tls.unpadding(frame, (frame['real_rows'],
#                                       frame['real_cols']))
#         if not cv2.imwrite(dest_path + str(c) + ".png",
#                            frame, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
#             print "Failed to create: " + dest_path + str(c) + ".png"
#     pickle.dump(header, open(dest_path + "header.dat", "w"))


def compress_error_fvspeck(path, dest_path, data_path, dec_path, dec_level):
    if path[-1] != "/":
        path += "/"
    info = pickle.load(open(path + "header.dat"))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not os.path.exists(dec_path):
        os.makedirs(dec_path)
    header = MainHeader()
    header.frames = info.frames
    header.ext = '.dat'
    codec = sk.ar_fvspeck()
    for c in range(224, info.frames):
        frame = np.load(path + str(c) + ".npy")
        h264data = pickle.load(open(data_path + str(c) + ".hdr"))
        rows = frame.shape[0]
        cols = frame.shape[1]
        frame = tls.zero_padding(frame)
        wavelet = wave.cdf97(frame)
        wavelet = tls.quantize(wavelet, 1000, dtype=int)
        bpp = h264data.abac_size / (rows * cols)
        print bpp
        center = (int(rows / 2), int(cols / 2))
        coded_frame = codec.compress(wavelet, bpp, 0.006, center, 0.3, 1, 1)
        coded_frame['real_cols'] = cols
        coded_frame['real_rows'] = rows
        wvlt = codec.clone
        iframe = wave.icdf97(wvlt)
        iframe = tls.quantize(iframe, 0.001)
        iframe = tls.unpadding(frame, (rows, cols))
        try:
            pickle.dump(coded_frame,
                        open(dest_path + str(c) + header.ext, "wb"))

            pickle.dump(iframe,
                        open(dec_path + str(c) + header.ext, "wb"))
        except:
            print "Failed to create: " + dest_path + str(c) + header.ext
    pickle.dump(header, open(dest_path + "header.dat", "w"))
    pickle.dump(header, open(dec_path + "header.dat", "w"))


def test_fvspeck(path, dest_path, dec_level):
    if path[-1] != "/":
        path += "/"
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    codec = sk.fv_speck()
    frame = cv2.imread(path + str(0) + ".png",
                       cv2.CV_LOAD_IMAGE_GRAYSCALE)
    rows = frame.shape[0]
    cols = frame.shape[1]
    frame = tls.zero_padding(frame)
    wavelet = wave.cdf97(frame, dec_level)
    wavelet = tls.quantize(wavelet, 1000, dtype=int)
    center = (int(512 / 2), int(512 / 2))
    coded_frame = codec.compress(wavelet, 3, 0.06, center, 0.3, 1, 40)
    coded_frame['real_cols'] = cols
    coded_frame['real_rows'] = rows
    wvlt = codec.clone
    iframe = wave.icdf97(wvlt)
    iframe = tls.quantize(iframe, 0.001)
    iframe = tls.unpadding(iframe, (rows, cols))
    iframe2 = tls.normalize(iframe, upper_bound=255, dtype=np.uint8)
    cv2.imwrite(dest_path + "21006_2.png", iframe2, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0])
    pickle.dump(iframe, open(dest_path + "21006_2.npy", "w"))
#second
    # coded_frame = codec.compress(wavelet, bpp, 0.006, center, 0.3, 1, 0.3)
    # coded_frame['real_cols'] = cols
    # coded_frame['real_rows'] = rows
    # wvlt = codec.clone
    # iframe = wave.icdf97(wvlt)
    # iframe = tls.quantize(iframe, 0.001)
    # iframe = tls.unpadding(frame, (rows, cols))
    # iframe2 = tls.normalize(iframe, upper_bound=255, dtype=np.uint8)
    # if not cv2.imwrite(dec_path + str(c) + "03.png",
    #                     iframe2, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
    #     print "Failed to create: " + dest_path + str(c) + ".png"
    # pickle.dump(iframe, open(dest_path +str(c) + ".npy","w"))


def test_speck(path, dest_path, dec_level):
    if path[-1] != "/":
        path += "/"
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    codec = sk.speck()
    frame = cv2.imread(path + str(0) + ".png",
                       cv2.CV_LOAD_IMAGE_GRAYSCALE)
    rows = frame.shape[0]
    cols = frame.shape[1]
    frame = tls.zero_padding(frame)
    wavelet = wave.cdf97(frame, dec_level)
    wavelet = tls.quantize(wavelet, 1000, dtype=int)
    center = (int(rows / 2), int(cols / 2))
    coded_frame = codec.compress(wavelet, 3)
    coded_frame['real_cols'] = cols
    coded_frame['real_rows'] = rows
    wvlt = codec.clone
    iframe = wave.icdf97(wvlt)
    iframe = tls.quantize(iframe, 0.001)
    iframe = tls.unpadding(iframe, (rows, cols))
    iframe2 = tls.normalize(iframe, upper_bound=255, dtype=np.uint8)
    cv2.imwrite(dest_path + "plain5_2a.png", iframe2, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0])
    pickle.dump(iframe, open(dest_path + "plain5_2.npy", "w"))


# def decompress_error_fvspeck(path, dest_path, data_path, dec_level):
#     if path[-1] != "/":
#         path += "/"
#     info = pickle.load(open(path + "header.dat"))
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#     header = MainHeader()
#     header.frames = info.frames
#     header.ext = '.png'
#     codec = sk.ar_speck()
#     for c in range(1, info.frames):
#         r = pickle.load(open(path + str(c) + info.ext))
#         wavelet = codec.expand(r['payload'], r['colums'], r['rows'], r['level'],
#                                r['wisebit'], r['Lbpp'], r['bpp'], ['center'],
#                                r['alpha'], r['c'], r['gamma'])
#         frame = wave.icdf97(wavelet, dec_level)
#         frame = tls.quantize(frame, 0.001, dtype=int)
#         frame = tls.unpadding(frame, (frame['real_rows'],
#                                       frame['real_cols']))
#         if not cv2.imwrite(dest_path + str(c) + ".png",
#                            frame, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
#             print "Failed to create: " + dest_path + str(c) + ".png"
#     pickle.dump(header, open(dest_path + "header.dat", "w"))


if __name__ == '__main__':
    # split_raw("/Users/juancgalan/Documents/video_test/akiyo/original/akiyo_cif.mov",
    #           "/Users/juancgalan/Documents/video_test/akiyo/raw/")
    # compress_fullsearch(
    #     "/Users/juancgalan/Documents/video_test/akiyo/raw/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/fullsearch/")
    # compress_key_h265(
    #     "/home/zenathar/Documents/video_test/akiyo/raw/",
    #     "/home/zenathar/Documents/video_test/akiyo/h265key/")
    # decompress_key_h265(
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265key/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265keydec/")
    # compress_error_h265(
    #     "/Users/juancgalan/Documents/video_test/akiyo/fullsearch/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265error/")
    # decompress_error_h265(
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265error/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265errordec/")
    # decompress_fullsearch(
    #     "/Users/juancgalan/Documents/video_test/akiyo/fullsearch/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/defullsearch/")
    # compress_key_speck(
    #     "/Users/juancgalan/Documents/video_test/akiyo/raw/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/speckkey/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265key/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/despeckkey/",
    #     4)
    # decompress_key_speck(
    #     "/Users/juancgalan/Documents/video_test/akiyo/speckkey/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/despeckkey/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265key/",
    #     4)
    # compress_key_fvspeck(
    #     "/Users/juancgalan/Documents/video_test/akiyo/raw/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/fvspeckkey/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265key/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/defvspeckkey/",
    #     4)
    # decompress_key_fvspeck(
    #     "/Users/juancgalan/Documents/video_test/akiyo/fvskpeckkey/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/defvspeckkey/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265key/",
    #     4)
    # compress_error_speck(
    #     "/Users/juancgalan/Documents/video_test/akiyo/fullsearch/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/speckerror/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265error/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/despeckerror/",
    #     4)
    # decompress_error_speck(
    #     "/Users/juancgalan/Documents/video_test/akiyo/speckerror/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/despeckerror/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265error/",
    #     4)
    # compress_error_fvspeck(
    #     "/home/zenathar/Documents/video_test/akiyo/fullsearch/",
    #     "/home/zenathar/Documents/video_test/akiyo/fvspeckerror/",
    #     "/home/zenathar/Documents/video_test/akiyo/h265error/",
    #     "/home/zenathar/Documents/video_test/akiyo/defvspeckerror/",
    #     4)
    # decompress_error_fvspeck(
    #     "/Users/juancgalan/Documents/video_test/akiyo/fvspeckerror/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/defvspeckerror/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265error/",
    #     4)
    # i = cv2.imread("/Users/juancgalan/Documents/video_test/akiyo/raw/0.png")
    # j = cv2.imread("/Users/juancgalan/Documents/video_test/akiyo/speckkey/0.png")
    test_fvspeck(
         "/home/zenathar/Documents/video_test/akiyo/raw/",
         "/home/zenathar/Documents/video_test/akiyo/testfvspeck/",
         4)
    test_speck(
         "/home/zenathar/Documents/video_test/akiyo/raw/",
         "/home/zenathar/Documents/video_test/akiyo/testfvspeck/",
         4)
