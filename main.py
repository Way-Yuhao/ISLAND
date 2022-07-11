import time
import datetime
import os
import os.path as p
from multiprocessing import Manager, Pool
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from interpolator import Interpolator
from config import *


def plot_means_trend():
    interp_1 = Interpolator(root='./data/export/', target_date='20181205')
    interp_2 = Interpolator(root='./data/export/', target_date='20181221')
    plt.figure(figsize=(10, 5))
    i = 0
    for c, _ in NLCD_2019_META['lut'].items():
        c = int(c)
        temp_for_c = interp_1.target.copy()
        temp_for_c[interp_1.nlcd != c] = 0
        dp = temp_for_c[np.where(temp_for_c != 0)]
        if len(dp > 0):
            x = np.ones_like(dp) * i
            y_mean = np.average(dp)
            y_std = np.std(dp)
            # plt.scatter(x=x, y=dp, s=3, c='#' + NLCD_2019_META['lut'][str(c)],
            #             label=NLCD_2019_META['class_names'][str(c)])
            # plt.errorbar(x=i, y=y_mean, yerr=y_std, fmt='.', color='black', capsize=3)
            plt.scatter(x=i, y=y_mean, s=50, facecolors='#' + NLCD_2019_META['lut'][str(c)],
                        label=NLCD_2019_META['class_names'][str(c)])
            i += 1
    i = 0
    for c, _ in NLCD_2019_META['lut'].items():
        c = int(c)
        temp_for_c = interp_2.target.copy()
        temp_for_c[interp_2.nlcd != c] = 0
        dp = temp_for_c[np.where(temp_for_c != 0)]
        if len(dp > 0):
            x = np.ones_like(dp) * i
            y_mean = np.average(dp)
            y_std = np.std(dp)
            plt.scatter(x=i, y=y_mean, s=50, edgecolors='#' + NLCD_2019_META['lut'][str(c)],
                        facecolors='none')
            # plt.errorbar(x=i, y=y_mean, yerr=y_std, fmt='.', color='black', capsize=3)
            i += 1
    plt.xlabel('NLCD Landcover Class')
    plt.ylabel('Brightness Temperature (K)')
    # plt.ylim(265, 300)
    # plt.xticks([])
    plt.xticks(np.arange(i))
    # plt.xticks(np.arange(len(dp)))
    plt.title('Changes of mean per landcover class between two dates')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=1)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.show()


def main():
    # interp = Interpolator(root='./data/export/', target_date='20181221')
    interp = Interpolator(root='./data/export/', target_date='20181205')
    # fpath = p.join(interp.root, 'cirrus', 'LC08_cirrus_houston_20181018.tif')
    # fpath = p.join(interp.root, 'cirrus', 'LC08_cirrus_houston_20190903.tif')
    # fpath = p.join(interp.root, 'cirrus', 'LC08_cirrus_houston_20190311.tif')
    # interp.add_occlusion(fpath)

    # interp.fill_average()
    # interp.display_target(mode='occluded')
    # interp.calc_loss(print_=True)
    # t = interp.calc_avg_temp_for_class(c=11)
    # print(t)
    # interp.calc_temp_per_class()
    # interp.plot_scatter_class()

    # mean = np.mean(interp.target)
    # mean_img = np.ones_like(interp.target) * mean
    # interp.reconstructed_target = mean_img
    # interp.calc_loss(print_=True, metric='mae')
    # interp.calc_loss(print_=True, metric='mse')

    # interp.spatial_interp()
    # interp.calc_loss(print_=True)
    # interp.display_target(mode='error')
    # interp.display_target(mode='reconst')


if __name__ == '__main__':
    # main()
    plot_means_trend()
