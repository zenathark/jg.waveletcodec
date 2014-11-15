import pickle
import cv2
import waveletcodec.tools as tls
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import numpy as np
import os


def calculate_psnr():
    path_or = os.getcwd() + "/data/akiyo/raw/"
    path_h264 = os.getcwd() + "/data/akiyo/h265keydec/"
    path_speck = os.getcwd() + "/data/akiyo/despeckkey/"
    path_afv = os.getcwd() + "/data/akiyo/defvspeckkey/"
    h264_psnr = []
    speck_psnr = []
    afv_psnr = []
    for c in range(299):
        frame = cv2.imread(path_or + str(c) + ".png",
                           cv2.CV_LOAD_IMAGE_GRAYSCALE)
        h_frame = cv2.imread(path_h264 + str(c) + ".png",
                             cv2.CV_LOAD_IMAGE_GRAYSCALE)
        s_frame = pickle.load(open(path_speck + str(c) + ".npy"))
        v_frame = cv2.imread(path_afv + str(c) + ".png",
                             cv2.CV_LOAD_IMAGE_GRAYSCALE)
        h264_psnr.append(tls.psnr(frame, h_frame))
        speck_psnr.append(tls.psnr(frame, s_frame))
        afv_psnr.append(tls.psnr(frame, v_frame))
    dest_path = os.getcwd() + "/data/akiyo/psnr"
    pickle.dump(h264_psnr, open(dest_path + "/h264key.dat", "w"))
    pickle.dump(speck_psnr, open(dest_path + "/speckkey.dat", "w"))
    pickle.dump(speck_psnr, open(dest_path + "/afvkey.dat", "w"))
    return


def create_key_graph():
    dest_path = os.getcwd() + "/data/akiyo/psnr"
    h264_psnr = pickle.load(open(dest_path + "/h264key.dat"))
    speck_psnr = pickle.load(open(dest_path + "/speckkey.dat"))
    afv_psnr = pickle.load(open(dest_path + "/afvkey.dat"))
    base_line = [33.21] * len(h264_psnr)
    h264_line = plt.plot(range(0,299), h264_psnr, "k-")
    speck_line = plt.plot(range(0,299), speck_psnr, "k.")
    afv_line = plt.plot(range(0,299), afv_psnr, "k+")
    afv_line = plt.plot(range(0,299), base_line, "k--")
    plt.legend(('H264/HEVC iDCT', 'SP-CODEC', 'AWFV-CODEC', 'DC Mode'), 'right', bbox_to_anchor=(1, 0.7 ))
    plt.grid(True)
    plt.xlabel('Frame index')
    plt.ylabel('PSNR (more is better)')
    plt.show()


def calculate_mse():
    path_or = "/home/zenathar/Documents/video_test/akiyo/fullsearch/"
    path_h264 = "/home/zenathar/Documents/video_test/akiyo/h265errordec/"
    path_speck = "/home/zenathar/Documents/video_test/akiyo/despeckerror/"
    path_afv = "/home/zenathar/Documents/video_test/akiyo/defvspeckerror/"
    h264_psnr = []
    speck_psnr = []
    afv_psnr = []
    for c in range(1, 299):
        frame = np.load(open(path_or + str(c) + ".npy"))
        h_frame = np.load(open(path_h264 + str(c) + ".npy"))
        s_frame = pickle.load(open(path_speck + str(c) + ".npy"))
        v_frame = pickle.load(open(path_afv + str(c) + ".npy"))
        h264_psnr.append(tls.mse(frame, h_frame))
        speck_psnr.append(tls.mse(frame, s_frame))
        afv_psnr.append(tls.mse(frame, v_frame))
    dest_path = "/home/zenathar/Documents/video_test/akiyo/psnr/"
    pickle.dump(h264_psnr, open(dest_path + "/h264error.dat", "w"))
    pickle.dump(speck_psnr, open(dest_path + "/speckerror.dat", "w"))
    pickle.dump(afv_psnr, open(dest_path + "/afverror.dat", "w"))
    return


def create_error_graph():
    dest_path = "/home/zenathar/Documents/video_test/akiyo/psnr"
    h264_psnr = pickle.load(open(dest_path + "/h264error.dat"))
    speck_psnr = pickle.load(open(dest_path + "/speckerror.dat"))
    afv_psnr = pickle.load(open(dest_path + "/afverror.dat"))
    h264_line = plt.plot(range(1,299), h264_psnr, "k-")
    speck_line = plt.plot(range(1,299), speck_psnr, "k+")
    afv_line = plt.plot(range(1,299), afv_psnr, "k*")
    plt.legend(('H264/HEVC iDCT', 'SP-Codec', 'AWFV-Codec'), 'right')
    plt.grid(True)
    plt.xlabel('Frame index')
    plt.ylabel('MSE (less is better)')
    plt.show()


def create_psnr_fovea():
    dest_path = os.getcwd() + "/data/akiyo"
    h264_psnr = pickle.load(open(dest_path + "/h264.npy"))
    speck_psnr = pickle.load(open(dest_path + "/speck.npy"))
    afv_psnr = pickle.load(open(dest_path + "/afv.npy"))
    plt.text(0, 0.1, 'End of fovea')
    h264_line = plt.plot(range(2,144)[:90], h264_psnr[:90], "k.")
    speck_line = plt.plot(range(2,144)[:90], speck_psnr[:90], "k*")
    afv_line = plt.plot(range(2,144)[:90], afv_psnr[:90], "k+")
    plt.axvline(x=75, color="k")
    plt.legend(('H.264', 'SPECK', 'AFV-SPECK'), 'right')
    plt.grid(True)
    plt.xlabel('Sub image size')
    plt.ylabel('PSNR (more is better)')
    plt.show()



if __name__ == "__main__":
    # calculate_psnr()
    # create_key_graph()
    # calculate_mse()
    # create_error_graph()
    create_psnr_fovea()
