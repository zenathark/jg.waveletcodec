from __future__ import division
import pickle
import cv2
import waveletcodec.tools as tls
import os


# output =  "/Users/juancgalan/Documents/spjour/"
# afvpath =  "/Users/juancgalan/Documents/spjour/img/21006.npy"
# speckpath =  "/Users/juancgalan/Documents/spjour/img/plain5.npy"
# orpath =  "/Users/juancgalan/Documents/spjour/img/0.png"
output = os.getcwd() + "/data/akiyo/"
afvpath = os.getcwd() + "/data/akiyo/ar21006_2.npy"
speckpath = os.getcwd() + "/data/akiyo/arplain5_2.npy"
h264path = os.getcwd() + "/data/akiyo/0_h264.png"
orpath = os.getcwd() + "/data/akiyo/0.png"
print orpath
original = cv2.imread(orpath, cv2.IMREAD_GRAYSCALE)
h264 = cv2.imread(h264path, cv2.IMREAD_GRAYSCALE)
afv = pickle.load(open(afvpath))
speck = pickle.load(open(speckpath))
cy = original.shape[0] // 2
cx = original.shape[1] // 2
sp = []
af = []
h2 = []
for i in range(2, cy):
    print i
    subimor = original[cy-i:cy+i,cx-i:cx+i]
    subimh2 = h264[cy-i:cy+i,cx-i:cx+i]
    subimspeck = speck[cy-i:cy+i,cx-i:cx+i]
    subimafv = afv[cy-i:cy+i,cx-i:cx+i]
    sp.append(tls.psnr(subimor, subimspeck))
    af.append(tls.psnr(subimor, subimafv))
    h2.append(tls.psnr(subimor, subimh2))
pickle.dump(sp, open(output + "speck.npy", "w"))
pickle.dump(af, open(output + "afv.npy", "w"))
pickle.dump(h2, open(output + "h264.npy", "w"))
