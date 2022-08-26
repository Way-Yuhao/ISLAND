import time
import datetime
import os
import os.path as p
from multiprocessing import Manager, Pool
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from interpolator import Interpolator
from config import *
from util.helper import deprecated
import textwrap


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
            print(y_mean)
            i += 1
    i = 0
    print('--------------------------------------------')
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
            print(y_mean)
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


def plot_per_pixel_diff():
    interp_1 = Interpolator(root='./data/export/', target_date='20181205')
    interp_2 = Interpolator(root='./data/export/', target_date='20181221')
    diff_img = interp_2.target - interp_1.target
    plt.figure(figsize=(10, 5))
    # sns.set(style='darkgrid')
    i = 0
    for c, _ in NLCD_2019_META['lut'].items():
        c = int(c)
        temp_for_c = diff_img.copy()
        temp_for_c[interp_1.nlcd != c] = 0
        dp = temp_for_c[np.where(temp_for_c != 0)]
        dp = dp[(dp > -100) & (dp < 100)]
        # filter out invalid pixels (delta greater than 100 Kelvin)

        if len(dp > 0):
            x = np.ones_like(dp) * i
            y_mean = np.average(dp)
            y_std = np.std(dp)
            plt.scatter(x=x, y=dp, s=3, c='#' + NLCD_2019_META['lut'][str(c)],
                        label=NLCD_2019_META['class_names'][str(c)])
            plt.errorbar(x=i, y=y_mean, yerr=y_std, fmt='.', color='black', capsize=3)
            i += 1
    plt.xlabel('NLCD Landcover Class')
    plt.ylabel('Brightness Temperature (K)')
    # plt.ylim(-10, 10)
    plt.xticks([])
    plt.title('Pixel-wise BT difference between two dates')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=5)
    plt.tight_layout()
    plt.show()


def swarm_per_pixel_diff():
    interp_1 = Interpolator(root='./data/export/', target_date='20181205')
    interp_2 = Interpolator(root='./data/export/', target_date='20181221')
    diff_img = interp_2.target - interp_1.target
    df = pd.DataFrame({'class': [], 'bt': []})
    palette = []
    plt.figure(figsize=(15, 5))
    sns.set(style='whitegrid')
    i = 0
    for c, _ in tqdm(NLCD_2019_META['lut'].items()):
        c = int(c)
        temp_for_c = diff_img.copy()
        temp_for_c[interp_1.nlcd != c] = 0
        dp = temp_for_c[np.where(temp_for_c != 0)]  # filter out pixels outside of current class
        # filter out invalid pixels (delta greater than 100 Kelvin)
        dp = dp[(dp > -100) & (dp < 100)]

        if len(dp > 0):
            x = len(dp) * [NLCD_2019_META['class_names'][str(c)]]
            new_df = pd.DataFrame({'class': x, 'bt': dp})
            df = pd.concat([df, new_df], ignore_index=True)
            palette += ['#' + NLCD_2019_META['lut'][str(c)]]
        i += 1
    ax = sns.violinplot(x='class', y='bt', data=df, palette=palette)
    ax.set_xticklabels(textwrap.fill(x.get_text(), 11) for x in ax.get_xticklabels())
    plt.xlabel('NLCD Landcover Class')
    plt.ylabel('Brightness Temperature (K)')
    plt.title('Pixel-wise BT difference between two dates')
    plt.tight_layout()
    plt.show()


def vis_per_pixel_diff():
    # interp_1 = Interpolator(root='./data/export/', target_date='20181205')
    # interp_2 = Interpolator(root='./data/export/', target_date='20181221')
    interp_1 = Interpolator(root='./data/Phoenix/', target_date='20200113')
    interp_2 = Interpolator(root='./data/Phoenix/', target_date='20190721')
    diff_img = interp_2.target - interp_1.target
    diff_img[diff_img > 100] = 0
    diff_img[diff_img < -100] = 0
    interp_1.display_target(img=diff_img, mode='error')

    i = 0
    for c, _ in tqdm(NLCD_2019_META['lut'].items()):
        c = int(c)
        temp_for_c = diff_img.copy()
        temp_for_c[interp_1.nlcd != c] = 0
        interp_1.display_target(img=temp_for_c, mode='error', text=NLCD_2019_META['class_names'][str(c)])
    #     # filter out invalid pixels (delta greater than 100 Kelvin)
    #     dp = dp[(dp > -100) & (dp < 100)]
    #
    #     if len(dp > 0):
    #         x = len(dp) * [NLCD_2019_META['class_names'][str(c)]]
    #         new_df = pd.DataFrame({'class': x, 'bt': dp})
    #         df = pd.concat([df, new_df], ignore_index=True)
    #         palette += ['#' + NLCD_2019_META['lut'][str(c)]]
    #     i += 1
    # ax = sns.violinplot(x='class', y='bt', data=df, palette=palette)
    # ax.set_xticklabels(textwrap.fill(x.get_text(), 11) for x in ax.get_xticklabels())
    # plt.xlabel('NLCD Landcover Class')
    # plt.ylabel('Brightness Temperature (K)')
    # plt.title('Pixel-wise BT difference between two dates')
    # plt.tight_layout()
    # plt.show()


def hist_per_pixel_diff():
    interp_1 = Interpolator(root='./data/export/', target_date='20181205')
    interp_2 = Interpolator(root='./data/export/', target_date='20181221')
    diff_img = interp_2.target - interp_1.target
    plt.figure(figsize=(10, 5))
    alpha, loc, beta = 5, 100, 22
    i = 0
    for c, _ in NLCD_2019_META['lut'].items():
        c = int(c)
        temp_for_c = diff_img.copy()
        temp_for_c[interp_1.nlcd != c] = 0
        dp = temp_for_c[np.where(temp_for_c != 0)]
        dp = dp[(dp > -100) & (dp < 100)]
        # filter out invalid pixels (delta greater than 100 Kelvin)

        if len(dp > 0):
            y_mean = np.average(dp)
            y_std = np.std(dp)
            # plt.scatter(x=x, y=dp, s=3, c='#' + NLCD_2019_META['lut'][str(c)],
            #             label=NLCD_2019_META['class_names'][str(c)])
            # plt.errorbar(x=i, y=y_mean, yerr=y_std, fmt='.', color='black', capsize=3)
            plt.hist(dp, bins=1000)
            plt.show()
            i += 1
    plt.xlabel('NLCD Landcover Class')
    plt.ylabel('Brightness Temperature (K)')
    # plt.ylim(-10, 10)
    plt.xticks([])
    plt.title('Pixel-wise BT difference between two dates')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=5)
    plt.tight_layout()
    plt.show()


def disp_per_pixel_diff():
    interp_1 = Interpolator(root='./data/export/', target_date='20181205')
    interp_2 = Interpolator(root='./data/export/', target_date='20181221')
    diff_img = interp_2.target - interp_1.target
    max_ = 10
    min_ = -10
    plt.imshow(diff_img, cmap='seismic', vmin=min_, vmax=max_)
    plt.title('Difference in BT between two dates.')
    plt.colorbar(label='BT(Kelvin)')
    plt.show()


def main_old():
    interp = Interpolator(root='./data/export/', target_date='20181221')

    img = interp.get_frame(frame_date='20181222', mode='cloud')
    # interp = Interpolator(root='./data/export/', target_date='20181205')
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
    # interp.plot_violins()

    # mean = np.mean(interp.target)
    # mean_img = np.ones_like(interp.target) * mean
    # interp.reconstructed_target = mean_img
    # interp.calc_loss(print_=True, metric='mae')
    # interp.calc_loss(print_=True, metric='mse')

    # interp.spatial_interp()
    # interp.calc_loss(print_=True)
    # interp.display_target(mode='error')
    # interp.display_target(mode='reconst')


def temp_eval_pairwise():
    interp = Interpolator(root='./data/export/', target_date='20181221')
    interp.occluded_target = interp.target.copy()
    # ref_frame_date = '20181205'
    # ref_frame_date = '20190327'
    ref_frame_date = '20180103'

    # no occlusion
    # interp.temporal_interp_as_is(ref_frame_date=ref_frame_date)
    # loss = interp.calc_loss(print_=True, metric='mae', entire_canvas=True)
    # interp.display_target(mode='error', text=f'no adjustment | MAE loss = {loss:.3f}')
    #
    # interp.temporal_interp_global_adj(ref_frame_date=ref_frame_date)
    # loss = interp.calc_loss(print_=True, metric='mae', entire_canvas=True)
    # interp.display_target(mode='error', text=f'global adjustment (class agnostic) | MAE loss = {loss:.3f}')

    interp.temporal_interp(ref_frame_date=ref_frame_date)
    # interp.temporal_interp_cloud(ref_frame_date=ref_frame_date, ref_syn_cloud_date='20180527')
    loss = interp.calc_loss(print_=True, metric='mae', entire_canvas=True)
    interp.display_target(mode='error', text=f'Ours temporal channel | MAE loss = {loss:.3f}')


def temp_multi_frame():
    interp = Interpolator(root='./data/Phoenix/', target_date='20220102')
    interp.add_occlusion(use_true_cloud=True)
    interp.temporal_interp_multi_frame(num_frames=3, max_delta_cycle=2, max_cloud_perc=.1)


def main():
    interp = Interpolator(root='./data/Phoenix/', target_date='20190822')
    # interp.temporal_interp()
    img = interp.build_valid_mask()
    plt.imshow(img)
    plt.show()
    print(img)


if __name__ == '__main__':
    # main()
    # plot_means_trend()
    # swarm_per_pixel_diff()
    # hist_per_pixel_diff()
    # vis_per_pixel_diff()
    # temp_eval_pairwise()
    temp_multi_frame()
