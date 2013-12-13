import cv2
import os
import pickle
import numpy as np
import waveletcodec.intraframe as intra
import waveletcodec.speck as sk
import waveletcodec.lwt as lwt
import waveletcodec.tools as tls
import waveletcodec.h264 as h264
import waveletcodec.entropy as etr


class MainHeader(object):
    frames = 0
    ext = ".png"


class IntraHeader(object):
    frames = 0
    ext = ".npy"
    search_size = 100
    block_size = 8


class HEVCHeader(object):
    frames = 0
    qstep = 1020
    ext = ".npy"
    shape = 0


class HEVCFrameHeader(object):
    abac_size = 0


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
#
#
# def decompress_fullsearch(path, dest_path, macroblock_size=8):
#     if path[-1] != "/":
#         path += "/"
#     info = pickle.load(open(path + "info.dat", "r"))
#     is_key = 0
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#     for c in range(info.frames):
#         error = cv2.imread(path + str(c) + ".png",
#                            cv2.CV_LOAD_IMAGE_GRAYSCALE)
#         if is_key == 0:
#             if not cv2.imwrite(dest_path + str(c) + ".png",
#                                error, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
#                 print "Failed to create: " + dest_path + str(c) + ".png"
#             is_key = info.fixed_keyframe - 1
#             key_frame = error
#         else:
#             frame = intraframe.decode_motion_frame(error,
#                                                    info.motion_vectors[c],
#                                                    info.macroblock_size,
#                                                    key_frame)
#
#             if not cv2.imwrite(dest_path + str(c) + ".png",
#                                frame, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
#                 print "Failed to create: " + dest_path + str(c) + ".png"
#             is_key -= 1
#     pickle.dump(info, open(dest_path + "info.dat", "w"))
#
#
# def compress_speck(path, dest_path, dec_level, bpp):
#     if path[-1] != "/":
#         path += "/"
#     info = pickle.load(open(path + "info.dat", "r"))
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#     codec = sk.speck()
#     info.wavelet = "cdf97"
#     info.wavelet_level = dec_level
#     info.frames = 3
#     info.bpp = bpp
#     for c in range(info.frames):
#         frame = cv2.imread(path + str(c) + ".png",
#                                     cv2.CV_LOAD_IMAGE_GRAYSCALE)
#         info.cols = frame.shape[1]
#         info.rows = frame.shape[0]
#         frame = tools.zero_padding(frame)
#         info.wavelet_cols = frame.shape[1]
#         info.wavelet_rows = frame.shape[0]
#         wavelet = lwt.cdf97(frame, dec_level)
#         wavelet = tools.quant(wavelet, 0.00001)
#         coded_frame = codec.compress(wavelet, bpp)
#         stream = dict()
#         stream["wise_bit"] = coded_frame[3]
#         stream["payload"] = coded_frame[4]
#         try:
#             pickle.dump(stream, open(dest_path + str(c) + ".speck", "wb"))
#         except:
#             print "Failed to create: " + dest_path + str(c) + ".speck"
#     pickle.dump(info, open(dest_path + "info.dat", "w"))
#
#
# def decompress_speck(path, dest_path):
#     if path[-1] != "/":
#         path += "/"
#     info = pickle.load(open(path + "info.dat", "r"))
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#     codec = sk.speck()
#     for c in range(info.frames):
#         frame = pickle.load(open(path + str(c) + ".speck","rb"))
#         wavelet = codec.expand(frame["payload"], info.wavelet_cols,
#                                    info.wavelet_rows, info.wavelet_level,
#                                    frame["wise_bit"])
#         iframe = lwt.icdf97(wavelet)
#         iframe = tools.aquant(iframe, 100000)
#         iframe = tools.unpadding(iframe, (info.rows, info.cols))
#         if not cv2.imwrite(dest_path + str(c) + ".png",
#                             iframe, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0]):
#             print "Failed to create: " + dest_path + str(c) + ".png"
#     pickle.dump(info, open(dest_path + "info.dat", "w"))
#
# def compress_fvht(path, dest_path):
#     print "TODO"
#
#
# def compress_fvht_fullsearch(path, dest_path):
#     print "TODO"
#
#
# def compress_motion_speck(path, dest_path, bpp, dec_level=4,
#                           macroblock_size=8, fixed_keyframe=0):
#     """This method compress a image sequence from a directory using speck and
#     motion compensation.
#
#     Args:
#         path: The directory where the image sequence to be encoded is stored
#         dest_path: A destination directory where the encoded frames will be
#                    stored
#         bpp: Compression ratio on bits per pixel
#
#     Kwargs:
#         dec_level: Level of wavelet decomposition to be used
#         macroblock_size: size of the macroblock used for motion compensation
#         fixed_keyframe: size of the Group of Pictures
#
#     """
#     if path[-1] != "/":
#         path += "/"
#     info = pickle.load(open(path + "info.dat", "r"))
#     is_key = 0
#     info.fixed_keyframe = fixed_keyframe
#     info.full_size = 1
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#     info.macroblock_size = macroblock_size
#     codec = sk.speck()
#     info.wavelet = "cdf97"
#     info.wavelet_level = dec_level
#     info.full_size = 100
#     for c in range(info.frames):
#         original_frame = cv2.imread(path + str(c) + ".png",
#                                     cv2.CV_LOAD_IMAGE_GRAYSCALE)
#         info.cols = original_frame.shape[1]
#         info.rows = original_frame.shape[0]
#         frame = tools.zero_padding(original_frame)
#         info.wavelet_cols = frame.shape[1]
#         info.wavelet_rows = frame.shape[0]
#         if is_key == 0:
#             wavelet = lwt.cdf97(frame, dec_level)
#             wavelet = tools.quant(wavelet, 0.0001)
#             coded_frame = codec.compress(wavelet, bpp)
#             stream = dict()
#             stream["wise_bit"] = coded_frame[3]
#             stream["payload"] = coded_frame[4]
#             try:
#                 pickle.dump(stream, open(dest_path + str(c) + ".speck", "wb"))
#             except:
#                 print "Failed to create: " + dest_path + str(c) + ".png"
#             iwave = codec.expand(coded_frame[4], frame.shape[1],
#                                  frame.shape[0], dec_level, coded_frame[3])
#             iframe = lwt.icdf97(iwave)
#             is_key = fixed_keyframe - 1
#             key_frame = iframe
#             info.motion_vectors += [0]
#         else:
#             p_frame, mvs = intraframe.encode_motion_frame(frame,
#                                                           key_frame,
#                                                           macroblock_size,
#                                                           info.full_size)
#             info.motion_vectors += [(mvs)]
#             is_key -= 1
#             wavelet = lwt.cdf97(p_frame, dec_level)
#             wavelet = tools.quant(wavelet, 0.0001)
#             coded_frame = codec.compress(wavelet, bpp)
#             stream = dict()
#             stream["wise_bit"] = coded_frame[3]
#             stream["payload"] = coded_frame[4]
#             try:
#                 pickle.dump(stream, open(dest_path + str(c) + ".speck", "wb"))
#             except:
#                 print "Failed to create: " + dest_path + str(c) + ".png"
#     pickle.dump(info, open(dest_path + "info.dat", "w"))
#
# def compress_motion_speck(path, dest_path, bpp, dec_level=4,
#                           macroblock_size=8, fixed_keyframe=0):
#     """This method compress a image sequence from a directory using speck and
#     motion compensation.
#
#     Args:
#         path: The directory where the image sequence to be encoded is stored
#         dest_path: A destination directory where the encoded frames will be
#                    stored
#         bpp: Compression ratio on bits per pixel
#
#     Kwargs:
#         dec_level: Level of wavelet decomposition to be used
#         macroblock_size: size of the macroblock used for motion compensation
#         fixed_keyframe: size of the Group of Pictures
#
#     """
#     if path[-1] != "/":
#         path += "/"
#     info = pickle.load(open(path + "info.dat", "r"))
#     is_key = 0
#     info.fixed_keyframe = fixed_keyframe
#     info.full_size = 1
#     if not os.path.exists(dest_path):
#         os.makedirs(dest_path)
#     info.macroblock_size = macroblock_size
#     codec = sk.speck()
#     info.wavelet = "cdf97"
#     info.wavelet_level = dec_level
#     info.full_size = 100
#     for c in range(info.frames):
#         original_frame = cv2.imread(path + str(c) + ".png",
#                                     cv2.CV_LOAD_IMAGE_GRAYSCALE)
#         info.cols = original_frame.shape[1]
#         info.rows = original_frame.shape[0]
#         frame = tools.zero_padding(original_frame)
#         info.wavelet_cols = frame.shape[1]
#         info.wavelet_rows = frame.shape[0]
#         if is_key == 0:
#             wavelet = lwt.cdf97(frame, dec_level)
#             wavelet = tools.quant(wavelet, 0.0001)
#             coded_frame = codec.compress(wavelet, bpp)
#             stream = dict()
#             stream["wise_bit"] = coded_frame[3]
#             stream["payload"] = coded_frame[4]
#             try:
#                 pickle.dump(stream, open(dest_path + str(c) + ".speck", "wb"))
#             except:
#                 print "Failed to create: " + dest_path + str(c) + ".png"
#             iwave = codec.expand(coded_frame[4], frame.shape[1],
#                                  frame.shape[0], dec_level, coded_frame[3])
#             iframe = lwt.icdf97(iwave)
#             is_key = fixed_keyframe - 1
#             key_frame = iframe
#             info.motion_vectors += [0]
#         else:
#             p_frame, mvs = intraframe.encode_motion_frame(frame,
#                                                           key_frame,
#                                                           macroblock_size,
#                                                           info.full_size)
#             info.motion_vectors += [(mvs)]
#             is_key -= 1
#             wavelet = lwt.cdf97(p_frame, dec_level)
#             wavelet = tools.quant(wavelet, 0.0001)
#             coded_frame = codec.compress(wavelet, bpp)
#             stream = dict()
#             stream["wise_bit"] = coded_frame[3]
#             stream["payload"] = coded_frame[4]
#             try:
#                 pickle.dump(stream, open(dest_path + str(c) + ".speck", "wb"))
#             except:
#                 print "Failed to create: " + dest_path + str(c) + ".png"
#     pickle.dump(info, open(dest_path + "info.dat", "w"))
#
# def decompress_motion_speck(path, dest_path):
#
#     pass
#
# def compress_spiht_fullsearch(path, dest_path):
#     print "TODO"

if __name__ == '__main__':
     # split_raw("/Users/juancgalan/Documents/video_test/akiyo/original/akiyo_cif.mov",
     #           "/Users/juancgalan/Documents/video_test/akiyo/raw/")
    # compress_fullsearch(
    #     "/Users/juancgalan/Documents/video_test/akiyo/raw/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/fullsearch/")
    # compress_key_h265(
    #     "/Users/juancgalan/Documents/video_test/akiyo/raw/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265key/")
    # decompress_key_h265(
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265key/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265keydec/")
    # compress_error_h265(
    #     "/Users/juancgalan/Documents/video_test/akiyo/fullsearch/",
    #     "/Users/juancgalan/Documents/video_test/akiyo/h265error/")
    decompress_error_h265(
        "/Users/juancgalan/Documents/video_test/akiyo/h265error/",
        "/Users/juancgalan/Documents/video_test/akiyo/h265errordec/")
    # decompress_fullsearch(
    #     "/Users/juancgalan/Downloads/video_test/akiyo/fullsearch/",
    #     "/Users/juancgalan/Downloads/video_test/akiyo/defullsearch/")
    # compress_speck(
    #     "/Users/juancgalan/Downloads/video_test/akiyo/fullsearch/",
    #     "/Users/juancgalan/Downloads/video_test/akiyo/speck/",
    #     4,1)
    # decompress_speck(
    #     "/Users/juancgalan/Downloads/video_test/akiyo/speck/",
    #     "/Users/juancgalan/Downloads/video_test/akiyo/despeck/",

