from __future__ import division
import pickle
import cv2
import waveletcodec.tools as tls


# output =  "/Users/juancgalan/Documents/spjour/"
# afvpath =  "/Users/juancgalan/Documents/spjour/img/21006.npy"
# speckpath =  "/Users/juancgalan/Documents/spjour/img/plain5.npy"
# orpath =  "/Users/juancgalan/Documents/spjour/img/0.png"
output = "/home/zenathar/Documents/spjour/"
afvpath = "/home/zenathar/Documents/spjour/img/ar21006_2.npy"
speckpath = "/home/zenathar/Documents/spjour/img/arplain5_2.npy"
orpath = "/home/zenathar/Documents/spjour/img/0.png"
original = cv2.imread(orpath, cv2.IMREAD_GRAYSCALE)
afv = pickle.load(open(afvpath))
speck = pickle.load(open(speckpath))
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
    speck_line = plt.plot(range(0,299), speck_psnr, "k+")
    speck_line = plt.plot(range(0,299), speck_psnr, "k+")
    af.append(tls.psnr(subimor, subimafv))
pickle.dump(sp, open(output + "speck.npy", "w"))
pickle.dump(af, open(output + "afv.npy", "w"))
print size(sp)
print size(af)
