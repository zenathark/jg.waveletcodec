import cv2
import waveletcodec.tools as tl
import pickle
import os
import csv


__base_path = "test_db"
__original = ["/still_original/512/",
              "/videoframe_original/256/"]
__fullsearch = "/fullsearch/"
__speck_path = "/speck/"
__fvspeck_path = "/fvspeck/"
__h265 = "/h265/"
__output = "/analysis_data/"


def create_psnr_from_case(original_path, filename):
    img_original = cv2.imread(__base_path + original_path + filename,
                              cv2.IMREAD_GRAYSCALE)
    img_h265 = cv2.imread(__base_path + __h265 + filename,
                          cv2.IMREAD_GRAYSCALE)
    img_speck = cv2.imread(__base_path + __speck_path + filename,
                           cv2.IMREAD_GRAYSCALE)
    img_fvspeck = cv2.imread(__base_path + __fvspeck_path + filename,
                             cv2.IMREAD_GRAYSCALE)
    psnr_h265 = tl.psnr(img_original, img_h265)
    ssim_h265 = tl.ssim(img_original, img_h265)
    psnr_speck = tl.psnr(img_original, img_speck)
    ssim_speck = tl.ssim(img_original, img_speck)
    psnr_fvspeck = tl.psnr(img_original, img_fvspeck)
    ssim_fvspeck = tl.ssim(img_original, img_fvspeck)
    bpp_compression = pickle.load(open(__base_path +
                                       __h265 +
                                       filename
                                       + '.inf',
                                       "rb"))['bpp']
    return [filename.split('.')[0], bpp_compression, psnr_h265, ssim_h265, psnr_speck, ssim_speck, psnr_fvspeck, ssim_fvspeck]


def create_dir_psnr():
    db = [['Name', 'Compression Ratio (bpp)', 'h265 (psnr)', 'h265 (ssim)' 'SPECK (psnr)', 'SPECK (ssim)', 'FV-SPECK (psnr)', 'FV-SPECK (ssim)']]
    for opath in __original:
        for fimg in os.listdir(__base_path + opath):
            db.append(create_psnr_from_case(opath, fimg))
    with open(__base_path + __output + 'psnr.csv', 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
        wr.writerows(db)


if __name__ == "__main__":
    create_dir_psnr()
