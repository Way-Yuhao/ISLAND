import time
import datetime
import os
import os.path as p
from multiprocessing import Manager, Pool
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from interpolator import Interpolator
from natsort import natsorted
from tqdm import tqdm
import random
from config import *
from util.helper import get_season, rprint, yprint, time_func, pjoin


def read_npy_stack(path):
    vis_output_path = './data/Houston/vis_output'
    assert p.exists(path)
    if not p.exists(vis_output_path):
        os.mkdir(vis_output_path)
    elif len(os.listdir(vis_output_path)) != 0:
        raise FileExistsError(f'{vis_output_path} is not empty.')

    files = [f for f in os.listdir(path) if 'st.npy' in f and 'reconst' in f]
    # assume that files are sorted by date

    for f in tqdm(files):
        scene = np.load(pjoin(path, f))
        if np.mean(scene) < 10:
            print(f'Empty image {f} skipped')
        scene_avg_temp = np.average(scene)
        adj_scene = scene - scene_avg_temp
        print(scene_avg_temp)
        # plt.imshow(adj_scene, cmap='coolwarm', vmin=-10, vmax=10)
        # plt.title(f'Brightness Temperature on {f[9:17]} relative to image mean')
        # plt.colorbar(label='BT(Kelvin)')
        # output_filename = f'temp_wrt_mean_{f[9:17]}.png'
        # plt.savefig(pjoin(vis_output_path, output_filename))
        # plt.close()


def vis_heat(path):
    thres_celsius = 30
    thres_kelvin = thres_celsius + 273.15
    vis_output_path = './data/Houston/vis_output'
    assert p.exists(path)
    if not p.exists(vis_output_path):
        os.mkdir(vis_output_path)
    elif len(os.listdir(vis_output_path)) != 0:
        raise FileExistsError(f'{vis_output_path} is not empty.')

    files = [f for f in os.listdir(path) if 'st.npy' in f and 'reconst' in f]
    # assume that files are sorted by date

    for f in tqdm(files):
        scene = np.load(pjoin(path, f))
        if np.mean(scene) < 10:
            print(f'Empty image {f} skipped')
        hot_spots = scene - thres_kelvin
        hot_spots = hot_spots.clip(min=0)
        # print(scene_avg_temp)
        plt.imshow(hot_spots, cmap='coolwarm', vmin=-10, vmax=10)
        plt.title(f'Brightness Temperature on {f[9:17]} | hot spots above {thres_kelvin} K')
        plt.colorbar(label='BT(Kelvin)')
        output_filename = f'temp_wrt_mean_{f[9:17]}.png'
        plt.savefig(pjoin(vis_output_path, output_filename))
        plt.close()


def main():
    # read_npy_stack(path='data/Houston/output_timelapse/')
    vis_heat(path='data/Houston/output_timelapse/')


if __name__ == '__main__':
    main()
