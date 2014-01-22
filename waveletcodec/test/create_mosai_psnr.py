from __future__ import division
import pickle
import cv2
import waveletcodec.tools as tls


output = "/Users/juancgalan/Documents/spjour/"
afvpath = "/Users/juancgalan/Documents/spjour/img/21006.png"
speckpath = "/Users/juancgalan/Documents/spjour/img/plain5.png"
orpath = "/Users/juancgalan/Documents/spjour/img/0.png"
original = cv2.imread(orpath, cv2.IMREAD_GRAYSCALE)
afv = cv2.imread(afvpath, cv2.IMREAD_GRAYSCALE)
speck = cv2.imread(speckpath, cv2.IMREAD_GRAYSCALE)
cy = original.shape[0] // 2
cx = original.shape[1] // 2
sp = []
af = []
for i in range(2, cy):
    print i
    subimor = original[cy-i:cy+i,cx-i:cx+i]
    subimspeck = speck[cy-i:cy+i,cx-i:cx+i]
    subimafv = afv[cy-i:cy+i,cx-i:cx+i]
    sp.append(tls.psnr(subimor, subimspeck))
    af.append(tls.psnr(subimor, subimafv))
pickle.dump(sp, open(output + "speck.npy", "w"))
pickle.dump(af, open(output + "afv.npy", "w"))
