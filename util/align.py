from matplotlib import pyplot as plt
import cv2
import numpy
import matplotlib as mpl


def read(img_path):
    img = cv2.imread(img_path, -1)
    resized = cv2.resize(img, (2500, 2560), interpolation=cv2.INTER_AREA)
    return resized


def main():
    mpl.use('macosx')
    nlcd = read('../data/nlcd2.tif')
    thermal = read('../data/landsat.tif')
    # plt.imshow(nlcd)
    # # plt.imshow(thermal)
    # plt.show()
    cv2.imwrite('../data/nlcd2_resized.tif', nlcd)
    cv2.imwrite('../data/landsat_resized.tif', thermal)


if __name__ == '__main__':
    main()