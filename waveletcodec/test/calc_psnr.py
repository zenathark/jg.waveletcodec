import pickle
import cv2
import waveletcodec.tools as tls
import matplotlib.pyplot as plt
import numpy as np


def calculate_psnr():
    path_or = "/home/zenathar/Documents/video_test/akiyo/raw/"
    path_h264 = "/home/zenathar/Documents/video_test/akiyo/h265keydec/"
    path_speck = "/home/zenathar/Documents/video_test/akiyo/despeckkey/"
    h264_psnr = []
    speck_psnr = []
    for c in range(299):
        frame = cv2.imread(path_or + str(c) + ".png",
                           cv2.CV_LOAD_IMAGE_GRAYSCALE)
        h_frame = cv2.imread(path_h264 + str(c) + ".png",
                             cv2.CV_LOAD_IMAGE_GRAYSCALE)
        s_frame = pickle.load(open(path_speck + str(c) + ".npy"))
        h264_psnr.append(tls.psnr(frame, h_frame))
        speck_psnr.append(tls.psnr(frame, s_frame))
    dest_path = "/home/zenathar/Documents/video_test/akiyo/psnr/"
    pickle.dump(h264_psnr, open(dest_path + "/h264key.dat", "w"))
    pickle.dump(speck_psnr, open(dest_path + "/speckkey.dat", "w"))
    return


def create_key_graph():
    dest_path = "/home/zenathar/Documents/video_test/akiyo/psnr/"
    h264_psnr = pickle.load(open(dest_path + "/h264key.dat"))
    speck_psnr = pickle.load(open(dest_path + "/speckkey.dat"))
    h264_line = plt.plot(range(0,299), h264_psnr, "k-")
    speck_line = plt.plot(range(0,299), speck_psnr, "k+")
    speck_line = plt.plot(range(0,299), speck_psnr, "k+")
    plt.legend(('H264/HEVC iDCT', 'SPECK and LWT'), 'right')
    plt.grid(True)
    plt.xlabel('Frame index')
    plt.ylabel('PSNR (more is better)')
    plt.show()


def calculate_mse():
    path_or = "/home/zenathar/Documents/video_test/akiyo/fullsearch/"
    path_h264 = "/home/zenathar/Documents/video_test/akiyo/h265errordec/"
    path_speck = "/home/zenathar/Documents/video_test/akiyo/despeckerror/"
    h264_psnr = []
    speck_psnr = []
    for c in range(1, 299):
        frame = np.load(open(path_or + str(c) + ".npy"))
        h_frame = np.load(open(path_h264 + str(c) + ".npy"))
        s_frame = pickle.load(open(path_speck + str(c) + ".npy"))
        h264_psnr.append(tls.mse(frame, h_frame))
        speck_psnr.append(tls.mse(frame, s_frame))
    dest_path = "/home/zenathar/Documents/video_test/akiyo/psnr/"
    pickle.dump(h264_psnr, open(dest_path + "/h264error.dat", "w"))
    pickle.dump(speck_psnr, open(dest_path + "/speckerror.dat", "w"))
    return


def create_error_graph():
    dest_path = "/home/zenathar/Documents/video_test/akiyo/psnr/"
    h264_psnr = pickle.load(open(dest_path + "/h264error.dat"))
    speck_psnr = pickle.load(open(dest_path + "/speckerror.dat"))
    h264_line = plt.plot(range(1,299), h264_psnr, "k-")
    speck_line = plt.plot(range(1,299), speck_psnr, "k+")
    speck_line = plt.plot(range(1,299), speck_psnr, "k+")
    plt.legend(('H264/HEVC iDCT', 'SPECK and LWT'), 'right')
    plt.grid(True)
    plt.xlabel('Frame index')
    plt.ylabel('MSE (less is better)')
    plt.show()


def create_psnr_fovea():
    path_or = "/home/zenathar/Documents/video_test/akiyo/raw/0.png"
    path_speck = "/home/zenathar/Documents/video_test/akiyo/testfvspeck/21006.npy"
    path_fovea = "/home/zenathar/Documents/video_test/akiyo/testfvspeck/plain5.npy"
    frame = cv2.imread(path_or,
                       cv2.CV_LOAD_IMAGE_GRAYSCALE)
    framespeck = pickle.load(open(path_speck))
    framefovea = pickle.load(open(path_fovea))
    rows = frame.shape[0]
    cols = frame.shape[1]
    r, c = (int(rows / 2), int(cols / 2))
    print tls.psnr(frame, framespeck)
    print tls.psnr(frame, framefovea)
    print tls.psnr(frame[r-10:r+10, c-10: c+10], framespeck[r-10:r+10, c-10: c+10])
    print tls.psnr(frame[r-10:r+10, c-10: c+10], framefovea[r-10:r+10, c-10: c+10])




if __name__ == "__main__":
    # calculate_psnr()
    # create_key_graph()
    # calculate_mse()
    # create_error_graph()
    create_psnr_fovea()
