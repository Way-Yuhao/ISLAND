import os
from matplotlib import pyplot as plt
import numpy as np
import cv2


def cmap(img_path):
    if '.npy' in img_path:
        img = np.load(img_path)
    else:
        img = cv2.imread(img_path, -1)
    plt.imshow(img, cmap='magma', vmin=290, vmax=310)
    plt.axis('off')
    plt.savefig('/Users/yuhaoliu/Downloads/test.png', bbox_inches='tight')

def cmap_with_occ():
    img_path = '/Users/yuhaoliu/Downloads/LC08_B10_20210924.tif'
    occlusion_path = '/Users/yuhaoliu/Downloads/syn_occlusion_20210924.npy'
    if '.npy' in img_path:
        img = np.load(img_path)
    else:
        img = cv2.imread(img_path, -1)
    occlusion = np.load(occlusion_path)
    img[occlusion == 1] = 0
    plt.imshow(img, cmap='magma', vmin=290, vmax=310)
    plt.axis('off')
    plt.savefig('/Users/yuhaoliu/Downloads/test.png', bbox_inches='tight')


def print_size(img_path):
    if '.npy' in img_path:
        img = np.load(img_path)
    else:
        img = cv2.imread(img_path, -1)
    print(img.shape)


if __name__ == '__main__':
    # cmap('/Users/yuhaoliu/Downloads/reconst_20210924_st.npy')
    # cmap('/Users/yuhaoliu/Downloads/LC08_B10_20170909.tif')
    # cmap_with_occ()
    print_size('/Users/yuhaoliu/Downloads/ny_island.npy')