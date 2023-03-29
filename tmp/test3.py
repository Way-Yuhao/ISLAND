import cv2
from util.helper import save_cmap
import os.path as p
import numpy
from matplotlib import pyplot as plt


# theta = '0.09'
# input_path = f'../data/Houston/analysis/occlusion_progression_20180103/r_occlusion{theta}.tif'
# assert p.exists
# img = cv2.imread(input_path, -1)
# # img = np.load()
# assert img is not None
# out_path = f'../data/general/r_occlusion_houston_{theta}.png'
# save_cmap(img, out_path, palette='inferno', vmin=290, vmax=330)

img = cv2.imread('../data/Houston/bt_series/LC08_B10_20180103.tif', -1)
out_path = '../data/general/houston_input_cmap_20180103.png'
save_cmap(img, out_path, palette='inferno', vmin=270, vmax=290)
# plt.imshow(img)
# plt.show()