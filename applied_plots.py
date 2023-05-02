"""
Generates plots to be used in the paper
"""
import os
import os.path as p
from natsort import natsorted
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import shutil
from matplotlib import pyplot as plt
import cv2
import wandb
from datetime import date, timedelta, datetime
from rich.progress import track
from config import *
from interpolator import Interpolator
from util.helper import rprint, yprint, hash_, pjoin, save_cmap, get_season, deprecated
from util.geo_reference import save_geotiff
from util.occlusion_sampler import OcclusionSampler


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


def calc_avg_temp_per_class_over_time(city=""):
    """
    Generates a line graph of average temperature of a given land cover class over time,
    for a particular city.
    Assuming:
    * naming convention like reconst_t20170116_syn20170116_ref20191224_st.npy
    * static NLCD layer, inheriting the first frame
    :param city:
    :return:
    """
    # scanning files
    yprint(f'Building temperature trend per class for {city}...')
    timelapse_dir = f'./data/{city}/output_referenced/st/'
    assert p.exists(timelapse_dir)
    output_dir = f'./data/{city}/analysis/'
    if not p.exists(output_dir):
        os.mkdir(output_dir)
    print('saving outputs to ', output_dir)
    files = [f for f in os.listdir(timelapse_dir) if '.tif' in f]
    files = [f for f in files if 'aux' not in f]
    files = natsorted(files)
    print(f'Got {len(files)} files, with dates ranging from {files[0][3:11]} to {files[-1][3:11]}')
    f0 = files[0]
    # print(f0)
    interp0 = Interpolator(root=f'./data/{city}/', target_date=f0[3:11])
    nlcd = interp0.nlcd
    df = pd.DataFrame()
    for f in tqdm(files, desc='Scanning predicted frames'):
        new_row = {'date': f[3:11]}
        prediction = cv2.imread(p.join(timelapse_dir, f), -1)
        # all classes
        if not np.all(np.isnan(prediction)) and np.any(prediction):
            pred_copy = prediction.copy()
            new_row['all'] = [np.nanmean(pred_copy)]
            developed_area = np.zeros_like(prediction, dtype=bool)
            developed_area[nlcd == 21] = True
            developed_area[nlcd == 22] = True
            developed_area[nlcd == 23] = True
            developed_area[nlcd == 24] = True
            # built up
            temp_developed = pred_copy.copy()
            temp_developed[~developed_area] = 0
            px_count = np.count_nonzero(temp_developed)
            avg_temp_developed = np.sum(temp_developed) / px_count
            new_row['developed'] = [avg_temp_developed]
            # natural
            temp_natural = pred_copy.copy()
            temp_natural[developed_area] = 0
            px_count = np.count_nonzero(temp_natural)
            avg_temp_natural = np.sum(temp_natural) / px_count
            new_row['natural'] = [avg_temp_natural]
        else:
            new_row['all'] = [np.nan]
            new_row['developed'] = [np.nan]
            new_row['natural'] = [np.nan]
        # individual classes
        for c, _ in NLCD_2019_META['lut'].items():
            temp_for_c = prediction.copy()
            temp_for_c[nlcd != int(c)] = 0
            px_count = np.count_nonzero(temp_for_c)
            if px_count == 0:
                avg_temp = -1  # invalid
                # print('skipped')
            else:
                avg_temp = np.sum(temp_for_c) / px_count  # average temperature for class c on this day, scalar
                # print(f'average temp for class {c} is {avg_temp}')
            new_row[c] = [avg_temp]
        df_new_row = pd.DataFrame(new_row)
        if df.empty:
            df = df_new_row
        else:
            df = pd.concat([df, df_new_row], ignore_index=True)
    df.to_csv(p.join(output_dir, f'average_temp_trend_{hash_()}.csv'))
    print(f'Data frame saved to average_temp_trend_{hash_()}.csv')


def plot_avg_temp_per_class_over_time(city="", hash_code=None):
    """
    Generates a plot from .csv file produced by calc_avg_temp_per_class_over_time().
    :param city:
    :return:
    """
    sns.set(style='white', context='paper', font='Times New Roman', font_scale=1.5)
    output_dir = f'./data/{city}/analysis/'
    if not p.exists(output_dir):
        raise FileNotFoundError()
    files = os.listdir(output_dir)
    files = [f for f in files if 'average_temp_trend' in f]
    files = [f for f in files if 'csv' in f]
    files = [f for f in files if '._' not in f]  # neglect temporary files
    if len(files) == 0:
        raise FileNotFoundError('No corresponding csv files found in ', output_dir)
    elif len(files) > 1:
        if hash_code is None:
            raise FileExistsError('Multiple csv files found. Please specify a hashcode')
        else:  # hash code is specified
            files = [f for f in files if hash_code in f]
            if len(files) == 0:
                raise FileNotFoundError('No csv file matches with the specified hash code: ', hash_code)
            elif len(files) > 1:
                raise FileExistsError('Hash collision')

    # only 1 matching csv file exists
    assert(len(files) == 1)
    yprint(f'Parsing dataframe from {files[0]}')
    df = pd.read_csv(p.join(output_dir, files[0]))
    # clean up date

    # df.replace(-1, np.inf)
    df[df < 2] = np.nan
    plt.figure(figsize=(13, 8))
    x_dates = [datetime.strptime(str(date_str), '%Y%m%d') for date_str in df['date']]
    # palette = []
    for c, _ in NLCD_2019_META['lut'].items():
        # c = int(c)
        # if len([t for t in df[c] if t == -1]) > 3:
        #     print(f'land cover class {c} skipped.')
            # continue  # skip absent land cover classes
        # palette += ['#' + NLCD_2019_META['lut'][str(c)]]
        ax = sns.lineplot(data=df, x=x_dates, y=c, color='#' + NLCD_2019_META['lut'][str(c)],
                          label=NLCD_2019_META['class_names'][str(c)], markers=True)
    # plt.title(f'Mean brightness temperature of each land cover class for {city}')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=5)
    plt.legend(loc='lower center', markerscale=5, ncol=4, bbox_to_anchor=(0.5, -0.45), frameon=False)
    plt.ylabel('Surface Temperature (K)')
    plt.xlabel('Date')
    plt.tight_layout()
    # ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
    # plt.show()
    plt.savefig(f'./data/general/{city}_temp_trend.pdf')


def count_hotzones_freq_for(city='Houston', temp_type='st', threshold = 295):
    """
    Produce a map where the pixel values represent number of days that this location exceeds
    temperature threshold. Requires data to be store under .../city/output_referenced/
    :param city:
    :param temp_type: between 'st' for surface temperature and 'bt' for brightness temperature
    :return:
    """
    sns.set(style='whitegrid', context='paper', font='Times New Roman')
    if temp_type not in {'st', 'bt'}:
        raise NotImplementedError()
    else:
        timelapse_dir = f'./data/{city}/output_referenced/{temp_type}/'
        assert p.exists(timelapse_dir)
        files = natsorted(os.listdir(timelapse_dir))
        # f0 = np.load(p.join(timelapse_dir, files[0]))
        f0 = cv2.imread(p.join(timelapse_dir, files[0]), -1)
        aggregate = np.zeros_like(f0)
        if len(files) == 0:
            raise FileNotFoundError
        for f in tqdm(files):
            if '._' in f:
                continue
            # img = np.load(p.join(timelapse_dir, f))
            img = cv2.imread(p.join(timelapse_dir, f), -1)
            this_frame = np.zeros_like(f0)
            this_frame[img >= threshold] = 1
            aggregate += this_frame
        plt.imshow(aggregate, cmap='cividis', vmin=0, vmax=50)
        plt.colorbar(label=f'Number of day exceeding {threshold} Kelvin')
        plt.title(f'Hot zones '
                  f'from {files[0][3:11]} to {files[-1][3:11]} in {city}')
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        plt.tight_layout()

        if not p.exists(f'./data/{city}/analysis/'):
            os.mkdir(f'./data/{city}/analysis/')
        # np.save(f'./data/general/{city}_hotzones_{threshold}k.npy', aggregate)
        plt.savefig(f'./data/general/{city}_hotzones_{threshold}k.png', dpi=1000)
        save_geotiff(city, aggregate, files[0][3:11], out_path=f'./data/{city}/analysis/hotzones_{threshold}k.tif')
        plt.close()
##############################################################################

def how_performance_decreases_as_synthetic_occlusion_increases(city, date_):
    occlusion_size = 250
    num_occlusions = [2, 4, 8, 16, 24, 32, 48, 64, 80, 100]
    root_ = f'./data/{city}/'
    out_dir = p.join(root_, 'analysis', f'occlusion_progression_{date_}')
    log_fpath = p.join(out_dir, 'log.csv')
    if p.exists(p.join(root_, 'output')):
        raise FileExistsError('Output directory exists. Please rename the directory to preserve contents.')
    if not p.exists(p.join(root_, 'analysis')):
        os.mkdir(p.join(root_, 'analysis'))
    if not p.exists(out_dir):
        os.mkdir(out_dir)
    log = []
    # real occlusion (a minimal amount)
    interp = Interpolator(root_, date_)
    real_occlusion_perc = interp.add_occlusion(use_true_cloud=True)
    print('real occlusion % = ', real_occlusion_perc)
    interp.run_interpolation()
    output_file = f'./data/{city}/output/reconst_{date_}_st.npy'
    shutil.copyfile(output_file, p.join(out_dir, f'r_occlusion{real_occlusion_perc:.2f}%.npy'))
    shutil.rmtree(p.join(root_, 'output'))
    # synthetic occlusions
    for n in num_occlusions:
        interp = Interpolator(root_, date_)
        # interp.add_occlusion(use_true_cloud=True)
        added_occlusion = interp.add_random_occlusion(size=occlusion_size, num_occlusions=n)
        interp.run_interpolation()
        mae_loss, _ = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
        rmse_loss, _ = interp.calc_loss_hybrid(metric='rmse', synthetic_only_mask=added_occlusion)
        mse_loss, _ = interp.calc_loss_hybrid(metric='mse', synthetic_only_mask=added_occlusion)
        syn_occlusion_perc = np.count_nonzero(added_occlusion) / (added_occlusion.shape[0] * added_occlusion.shape[1])
        save_geotiff(city, interp.reconstructed_target, date_,
                     p.join(out_dir, f'r_occlusion{syn_occlusion_perc:.2f}.tif'))
        save_geotiff(city, added_occlusion.astype(float), date_,
                     p.join(out_dir, f'occlusion{syn_occlusion_perc:.2f}.tif'))
        # output_file = f'./data/{city}/output/reconst_{date_}_st.npy'
        # shutil.copyfile(output_file, p.join(out_dir, f'r_occlusion{syn_occlusion_perc:.2f}.npy'))
        log += [(syn_occlusion_perc, mae_loss, rmse_loss, mse_loss)]
    # add almost 100% cloudy frames
    # for n in reversed(num_occlusions[:3]):
    #     interp = Interpolator(root_, date_)
    #     temp_ = Interpolator(root_, date_)
    #     added_occlusion_negative = temp_.add_random_occlusion(size=occlusion_size, num_occlusions=n)
    #     added_occlusion = 1 - added_occlusion_negative
    #     del temp_
    #     interp.build_valid_mask()
    #     real_occlusion = interp.target_valid_mask
    #     interp.synthetic_occlusion = added_occlusion.copy()
    #     interp.occluded_target = interp.target.copy()
    #     interp.occluded_target[added_occlusion] = 0
    #     interp.run_interpolation()
    #     mae_loss, _ = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
    #     rmse_loss, _ = interp.calc_loss_hybrid(metric='rmse', synthetic_only_mask=added_occlusion)
    #     mse_loss, _ = interp.calc_loss_hybrid(metric='mse', synthetic_only_mask=added_occlusion)
    #     syn_occlusion_perc = np.count_nonzero(added_occlusion) / (added_occlusion.shape[0] * added_occlusion.shape[1])
    #     save_geotiff(city, interp.reconstructed_target, date_,
    #                  p.join(out_dir, f'r_occlusion{syn_occlusion_perc:.2f}.tif'))
    #     save_geotiff(city, added_occlusion.astype(float), date_,
    #                  p.join(out_dir, f'occlusion{syn_occlusion_perc:.2f}.tif'))
    #     log += [(syn_occlusion_perc, mae_loss, rmse_loss, mse_loss)]

    df = pd.DataFrame(log, columns=['syn_occlusion_perc', 'mae', 'rmse', 'mse'])
    df.to_csv(log_fpath, index=False)
    print(df)
    print('real occlusion % = ', real_occlusion_perc)
    shutil.rmtree(p.join(root_, 'output'))
    return


def find_dates_at_interval_theta(city, theta_intervals):
    select_idx = []
    df_path = f'./data/{city}/analysis/averages_by_date.csv'
    if not p.exists(df_path):
        root_ = f'./data/{city}'
        df = pd.read_csv(p.join(root_, 'metadata.csv'))
        dates = df['date'].values.tolist()
        dates = [str(d) for d in dates]
        log = []
        for d in tqdm(dates):
            interp = Interpolator(root=root_, target_date=d)
            theta = interp.add_occlusion(use_true_cloud=True)
            input_bitmask = np.array(~interp.synthetic_occlusion, dtype=np.bool_)
            input_bitmask[~interp.target_valid_mask] = False
            if np.any(input_bitmask):
                avg = np.average(interp.occluded_target[input_bitmask])
            else:
                avg = np.nan
            log += [(d, avg, theta)]
        df = pd.DataFrame(log, columns=['date', 'avg', 'theta'])
        df.to_csv(df_path, index=False)
        print('csv file saved to ', df_path)

    print('using csv file ', df_path)
    df = pd.read_csv(df_path)
    thetas = df['theta']
    for t in theta_intervals:
        idx = np.array([np.abs(t - theta) for theta in thetas]).argmin()
        select_idx.append(idx)
    selected_dates = [df['date'][i] for i in select_idx]
    return selected_dates

deprecated
def how_performance_decreases_as_synthetic_occlusion_increases2(city, date_, added_cloud_dates):
    """
    Instead of using random occlusion, we use real occlusions from another date.
    :param city:
    :param date_:
    :return:
    """
    # added_cloud_dates = [20180728, 20200514, 20200530, 20180813, 20220520, 20211227]
    # added_cloud_dates = [20211227]
    root_ = f'./data/{city}/'
    out_dir = p.join(root_, 'analysis', f'occlusion_progression_{date_}')
    log_fpath = p.join(out_dir, 'log.csv')
    if p.exists(p.join(root_, 'output')):
        raise FileExistsError('Output directory exists. Please rename the directory to preserve contents.')
    if not p.exists(p.join(root_, 'analysis')):
        os.mkdir(p.join(root_, 'analysis'))
    if not p.exists(out_dir):
        os.mkdir(out_dir)
    log = []
    # real occlusion (a minimal amount)
    interp = Interpolator(root_, date_)
    real_occlusion_perc = interp.add_occlusion(use_true_cloud=True)
    print('real occlusion % = ', real_occlusion_perc)
    interp.run_interpolation()
    output_file = f'./data/{city}/output/reconst_{date_}_st.npy'
    shutil.copyfile(output_file, p.join(out_dir, f'r_occlusion{real_occlusion_perc:.2f}%.npy'))
    shutil.rmtree(p.join(root_, 'output'))
    # synthetic occlusions
    for d in added_cloud_dates:
        interp = Interpolator(root_, date_)
        interp.add_occlusion(fpath=f'./data/{city}/cloud/LC08_cloud_{d}.tif')
        syn_occlusion = interp.synthetic_occlusion
        real_occlusion = ~interp.build_valid_mask()
        added_occlusion = syn_occlusion.copy()
        added_occlusion[real_occlusion] = False
        interp.run_interpolation()
        mae_loss, _ = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
        rmse_loss, _ = interp.calc_loss_hybrid(metric='rmse', synthetic_only_mask=added_occlusion)
        mse_loss, _ = interp.calc_loss_hybrid(metric='mse', synthetic_only_mask=added_occlusion)
        syn_occlusion_perc = np.count_nonzero(added_occlusion) / (added_occlusion.shape[0] * added_occlusion.shape[1])
        save_geotiff(city, interp.reconstructed_target, date_,
                     p.join(out_dir, f'r_occlusion{syn_occlusion_perc:.2f}.tif'))
        save_geotiff(city, added_occlusion.astype(float), date_,
                     p.join(out_dir, f'occlusion{syn_occlusion_perc:.2f}.tif'))
        # output_file = f'./data/{city}/output/reconst_{date_}_st.npy'
        # shutil.copyfile(output_file, p.join(out_dir, f'r_occlusion{syn_occlusion_perc:.2f}.npy'))
        log += [(syn_occlusion_perc, mae_loss, rmse_loss, mse_loss)]
    df = pd.DataFrame(log, columns=['syn_occlusion_perc', 'mae', 'rmse', 'mse'])
    df.to_csv(log_fpath, index=False)
    print(df)
    print('real occlusion % = ', real_occlusion_perc)
    shutil.rmtree(p.join(root_, 'output'))
    return


def how_performance_decreases_as_synthetic_occlusion_increases3(city, date_):
    theta_intervals = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.0]
    selected_dates = find_dates_at_interval_theta(city=city, theta_intervals=theta_intervals)
    root_ = f'./data/{city}/'
    out_dir = p.join(root_, 'analysis', f'occlusion_progression_{date_}_improved')
    log_fpath = p.join(out_dir, 'log.csv')
    if p.exists(p.join(root_, 'output')) and len(os.listdir(p.join(root_, 'output'))) > 1:
        raise FileExistsError('Output directory exists. Please rename the directory to preserve contents.')
    if not p.exists(p.join(root_, 'analysis')):
        os.mkdir(p.join(root_, 'analysis'))
    if not p.exists(out_dir):
        os.mkdir(out_dir)
    log = []
    # real occlusion (a minimal amount)
    interp = Interpolator(root_, date_)
    real_occlusion_perc = interp.add_occlusion(use_true_cloud=True)
    print('real occlusion % = ', real_occlusion_perc)
    # interp.run_interpolation()
    # output_file = f'./data/{city}/output/reconst_{date_}_st.npy'
    # shutil.copyfile(output_file, p.join(out_dir, f'r_occlusion{real_occlusion_perc:.2f}%.npy'))
    # shutil.rmtree(p.join(root_, 'output'))
    # synthetic occlusions
    for d in selected_dates:
        interp = Interpolator(root_, date_)
        real_occlusion = ~interp.build_valid_mask()
        # theta = interp.add_occlusion(fpath=f'./data/{city}/cloud/LC08_cloud_{d}.tif')
        # add occlusion
        cloud = cv2.imread(f'./data/{city}/cloud/LC08_cloud_{d}.tif', -1)
        shadow = cv2.imread(f'./data/{city}/shadow/LC08_shadow_{d}.tif', -1)
        occlusion = cloud + shadow
        occlusion[occlusion != 0] = 255
        interp.synthetic_occlusion = np.array(occlusion, dtype=np.bool_)  # FIXME
        interp.occluded_target = interp.target.copy()
        interp.occluded_target[occlusion] = 0
        px_count = occlusion.shape[0] * occlusion.shape[1]
        theta = np.count_nonzero(occlusion) / px_count
        # syn_occlusion = interp.synthetic_occlusion
        # real_occlusion = ~interp.build_valid_mask()
        added_occlusion = interp.synthetic_occlusion.copy()
        added_occlusion[real_occlusion] = False
        interp.run_interpolation()
        mae_loss, _ = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
        rmse_loss, _ = interp.calc_loss_hybrid(metric='rmse', synthetic_only_mask=added_occlusion)
        mse_loss, _ = interp.calc_loss_hybrid(metric='mse', synthetic_only_mask=added_occlusion)
        # syn_occlusion_perc = np.count_nonzero(added_occlusion) / (added_occlusion.shape[0] * added_occlusion.shape[1])
        save_geotiff(city, interp.reconstructed_target, date_,
                     p.join(out_dir, f'r_occlusion{theta:.2f}.tif'))
        save_geotiff(city, added_occlusion.astype(float), date_,
                     p.join(out_dir, f'occlusion{theta:.2f}.tif'))
        # output_file = f'./data/{city}/output/reconst_{date_}_st.npy'
        # shutil.copyfile(output_file, p.join(out_dir, f'r_occlusion{syn_occlusion_perc:.2f}.npy'))
        log += [(theta, mae_loss, rmse_loss, mse_loss)]
    df = pd.DataFrame(log, columns=['theta', 'mae', 'rmse', 'mse'])
    df.to_csv(log_fpath, index=False)
    print(df)
    print('real occlusion % = ', real_occlusion_perc)
    shutil.rmtree(p.join(root_, 'output'))
    return


def how_performance_decreases_as_synthetic_occlusion_increases4(city, date_):
    """
    Now testing on all occlusion factors
    :param city:
    :param date_:
    :return:
    """
    wandb.init()
    df_path = f'./data/{city}/analysis/averages_by_date.csv'
    if not p.exists(df_path):
        root_ = f'./data/{city}'
        df = pd.read_csv(p.join(root_, 'metadata.csv'))
        dates = df['date'].values.tolist()
        dates = [str(d) for d in dates]
        log = []
        for d in tqdm(dates):
            interp = Interpolator(root=root_, target_date=d)
            theta = interp.add_occlusion(use_true_cloud=True)
            input_bitmask = np.array(~interp.synthetic_occlusion, dtype=np.bool_)
            input_bitmask[~interp.target_valid_mask] = False
            if np.any(input_bitmask):
                avg = np.average(interp.occluded_target[input_bitmask])
            else:
                avg = np.nan
            log += [(d, avg, theta)]
        df = pd.DataFrame(log, columns=['date', 'avg', 'theta'])
        df.to_csv(df_path, index=False)
        print('csv file saved to ', df_path)
    df = pd.read_csv(df_path)
    root_ = f'./data/{city}/'
    out_dir = p.join(root_, 'analysis', f'occlusion_progression_{date_}_fixed_occ')
    log_fpath = p.join(out_dir, 'log.csv')
    if p.exists(p.join(root_, 'output')) and len(os.listdir(p.join(root_, 'output'))) > 1:
        raise FileExistsError('Output directory exists. Please rename the directory to preserve contents.')
    if not p.exists(p.join(root_, 'analysis')):
        os.mkdir(p.join(root_, 'analysis'))
    if not p.exists(out_dir):
        os.mkdir(out_dir)
    log = []
    # real occlusion (a minimal amount)
    interp = Interpolator(root_, date_)
    real_occlusion_perc = interp.add_occlusion(use_true_cloud=True)
    print('real occlusion % = ', real_occlusion_perc)

    for index, row in df.iterrows():
        # print(row['date'], row['theta'])
        d = str(int(row['date']))
        theta = row['theta']
        # if theta < 0.9 or theta > 0.99:
        #     continue
        interp = Interpolator(root_, date_)
        real_occlusion = ~interp.build_valid_mask()
        # theta = interp.add_occlusion(fpath=f'./data/{city}/cloud/LC08_cloud_{d}.tif')
        # add occlusion
        cloud = cv2.imread(f'./data/{city}/cloud/LC08_cloud_{d}.tif', -1)
        shadow = cv2.imread(f'./data/{city}/shadow/LC08_shadow_{d}.tif', -1)
        occlusion = cloud + shadow
        occlusion[occlusion != 0] = 255
        interp.synthetic_occlusion = np.array(occlusion, dtype=np.bool_)  # FIXME
        interp.occluded_target = interp.target.copy()
        interp.occluded_target[interp.synthetic_occlusion] = 0
        px_count = occlusion.shape[0] * occlusion.shape[1]
        theta = np.count_nonzero(occlusion) / px_count
        # syn_occlusion = interp.synthetic_occlusion
        # real_occlusion = ~interp.build_valid_mask()
        added_occlusion = interp.synthetic_occlusion.copy()
        added_occlusion[real_occlusion] = False
        interp.run_interpolation()
        mae_loss, _ = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
        rmse_loss, _ = interp.calc_loss_hybrid(metric='rmse', synthetic_only_mask=added_occlusion)
        mse_loss, _ = interp.calc_loss_hybrid(metric='mse', synthetic_only_mask=added_occlusion)
        mape_loss, _ = interp.calc_loss_hybrid(metric='mape', synthetic_only_mask=added_occlusion)
        # syn_occlusion_perc = np.count_nonzero(added_occlusion) / (added_occlusion.shape[0] * added_occlusion.shape[1])
        save_geotiff(city, interp.reconstructed_target, date_,
                     p.join(out_dir, f'r_occlusion{theta:.2f}.tif'))
        save_geotiff(city, added_occlusion.astype(float), date_,
                     p.join(out_dir, f'occlusion{theta:.2f}.tif'))
        # output_file = f'./data/{city}/output/reconst_{date_}_st.npy'
        # shutil.copyfile(output_file, p.join(out_dir, f'r_occlusion{syn_occlusion_perc:.2f}.npy'))
        log += [(theta, mae_loss, rmse_loss, mse_loss, mape_loss)]
    df = pd.DataFrame(log, columns=['theta', 'mae', 'rmse', 'mse', 'mape'])
    df.to_csv(log_fpath, index=False)
    print(df)
    print('real occlusion % = ', real_occlusion_perc)
    shutil.rmtree(p.join(root_, 'output'))
    wandb.alert(
        title='Process finished',
        text=f'Data for region {city} finished processing.'
    )
    return


def how_performance_decreases_as_synthetic_occlusion_increases5(city, date_, sampler):
    """
    Now with data augmentation
    :param city:
    :param date_:
    :return:
    """
    wandb.init()
    SAMPLES_PER_BIN = 10
    df_path = f'./data/{city}/analysis/averages_by_date.csv'
    if not p.exists(df_path):
        root_ = f'./data/{city}'
        df = pd.read_csv(p.join(root_, 'metadata.csv'))
        dates = df['date'].values.tolist()
        dates = [str(d) for d in dates]
        log = []
        for d in tqdm(dates):
            interp = Interpolator(root=root_, target_date=d)
            theta = interp.add_occlusion(use_true_cloud=True)
            input_bitmask = np.array(~interp.synthetic_occlusion, dtype=np.bool_)
            input_bitmask[~interp.target_valid_mask] = False
            if np.any(input_bitmask):
                avg = np.average(interp.occluded_target[input_bitmask])
            else:
                avg = np.nan
            log += [(d, avg, theta)]
        df = pd.DataFrame(log, columns=['date', 'avg', 'theta'])
        df.to_csv(df_path, index=False)
        print('csv file saved to ', df_path)
    df = pd.read_csv(df_path)
    root_ = f'./data/{city}/'
    out_dir = p.join(root_, 'analysis', f'occlusion_progression_{date_}_sample')
    log_fpath = p.join(out_dir, 'log.csv')
    if p.exists(p.join(root_, 'output')) and len(os.listdir(p.join(root_, 'output'))) > 1:
        raise FileExistsError('Output directory exists. Please rename the directory to preserve contents.')
    if not p.exists(p.join(root_, 'analysis')):
        os.mkdir(p.join(root_, 'analysis'))
    if not p.exists(out_dir):
        os.mkdir(out_dir)
    log = []
    # real occlusion (a minimal amount)
    interp = Interpolator(root_, date_)
    real_occlusion_perc = interp.add_occlusion(use_true_cloud=True)
    occlusion_shape = interp.synthetic_occlusion.shape
    print('real occlusion % = ', real_occlusion_perc)
    ranges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for theta_range in ranges:
        for k in range(SAMPLES_PER_BIN):
            # d = str(int(row['date']))
            # # theta = row['theta']
            interp = Interpolator(root_, date_)
            real_occlusion = ~interp.build_valid_mask()
            # # theta = interp.add_occlusion(fpath=f'./data/{city}/cloud/LC08_cloud_{d}.tif')
            # # add occlusion
            # cloud = cv2.imread(f'./data/{city}/cloud/LC08_cloud_{d}.tif', -1)
            # shadow = cv2.imread(f'./data/{city}/shadow/LC08_shadow_{d}.tif', -1)
            # occlusion = cloud + shadow
            # occlusion[occlusion != 0] = 255
            # interp.synthetic_occlusion = np.array(occlusion, dtype=np.bool_)  # FIXME
            occlusion = sampler.sample(theta_range, occlusion_shape)
            interp.synthetic_occlusion = occlusion
            interp.occluded_target = interp.target.copy()
            interp.occluded_target[interp.synthetic_occlusion] = 0
            px_count = occlusion.shape[0] * occlusion.shape[1]
            theta = np.count_nonzero(occlusion) / px_count
            added_occlusion = interp.synthetic_occlusion.copy()
            added_occlusion[real_occlusion] = False
            interp.run_interpolation()
            mae_loss, _ = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
            rmse_loss, _ = interp.calc_loss_hybrid(metric='rmse', synthetic_only_mask=added_occlusion)
            mse_loss, _ = interp.calc_loss_hybrid(metric='mse', synthetic_only_mask=added_occlusion)
            mape_loss, _ = interp.calc_loss_hybrid(metric='mape', synthetic_only_mask=added_occlusion)
            save_geotiff(city, interp.reconstructed_target, date_,
                         p.join(out_dir, f'r_occlusion{theta:.2f}.tif'))
            save_geotiff(city, added_occlusion.astype(float), date_,
                         p.join(out_dir, f'occlusion{theta:.2f}.tif'))
            log += [(theta, mae_loss, rmse_loss, mse_loss, mape_loss)]
    df = pd.DataFrame(log, columns=['theta', 'mae', 'rmse', 'mse', 'mape'])
    df.to_csv(log_fpath, index=False)
    print(df)
    print('real occlusion % = ', real_occlusion_perc)
    shutil.rmtree(p.join(root_, 'output'))
    wandb.alert(
        title='Process finished',
        text=f'Data for region {city} finished processing.'
    )
    return

def recalculate_degradation_error(city, date_):
    root_ = f'./data/{city}/'
    out_dir = p.join(root_, 'analysis', f'occlusion_progression_{date_}_improved')
    assert p.exists(out_dir)
    df_path = f'./data/{city}/analysis/averages_by_date.csv'
    df = pd.read_csv(df_path)
    for index, row in df.iterrows():
        d = str(int(row['date']))
        theta = row['theta']


def performance_degradation_graph(data_list):
    """
    line plot with dots
    :param data_list:
    :return:
    """
    sns.set_theme(style='white', context='paper', font='Times New Roman', font_scale=1.5)
    # log_path = './data/general/performance_degradation.csv'
    df = pd.DataFrame()
    for entry in data_list:
        city, date_ = entry[0], entry[1]
        log_path = f'./data/{city}/analysis/occlusion_progression_{date_}_island/log.csv'
        if not p.exists(log_path):
            rprint(f'File for {city} on {date_} does not exist.')
            continue
        current_df = pd.read_csv(log_path)
        current_df['city'] = city
        # print(df)
        df = pd.concat([df, current_df], ignore_index=True)
    # print(df)
    plot = sns.lineplot(data=df, y='mae', x='theta', hue='city', marker='o')
    plot.set_ylim(0.45, 1.5)
    plt.xlabel('Occlusion factor, \u03B8')
    plt.ylabel('MAE (K)')
    plt.legend(loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.22), frameon=False)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./data/general/degradation_plot.pdf')


def performance_degradation_graph2(data_list, y_axis_metric, hue=True):
    def categorize(row):
        theta = row['theta']
        if theta < 0.9:
            cat = int(theta * 10)/10  # take floor
        elif theta < 0.99:
            cat = 0.9
        else:
            cat = 1.0
        return cat
    if y_axis_metric not in ['mae', 'mape']:
        raise AttributeError()
    sns.set_theme(style='white', context='paper', font='Times New Roman', font_scale=1.5)
    # log_path = './data/general/performance_degradation.csv'
    df = pd.DataFrame()
    for entry in data_list:
        city, date_ = entry[0], entry[1]
        log_path = f'./data/{city}/analysis/occlusion_progression_{date_}_sample/log.csv'
        if not p.exists(log_path):
            rprint(f'File for {city} on {date_} does not exist.')
            continue
        current_df = pd.read_csv(log_path)
        current_df['city'] = city
        # print(df)
        df = pd.concat([df, current_df], ignore_index=True)
    # print(df)
    df['range'] = df.apply(lambda row: categorize(row), axis=1)
    # plot = sns.lmplot(data=df, y='mae', x='theta', hue='city')
    if y_axis_metric == 'mape':
        plt.figure(figsize=(6.4, 3))
    else:
        pass
    plot = sns.boxplot(data=df, y=y_axis_metric, x='range', color='lavender', showfliers=False)
    if hue is True:
        color = None
        hue = 'city'
    else:
        color = 'black'
        hue = None
    sns.stripplot(data=df, y=y_axis_metric, x='range', color=color, marker='o', hue=hue)
    # plot.set_xlim(-0.7, 9.7)
    if y_axis_metric == 'mae':
        plot.set_ylim(0.5, 3.6)
        plt.ylabel('MAE (K)')
        # plt.legend(loc='lower center', ncols=4, bbox_to_anchor=(0.5, -0.3), frameon=False,
        #            markerscale=0.8, columnspacing=0.7, handletextpad=0.2)
        plot.get_legend().remove()
    else:
        plot.set_ylim(0, 0.011)
        plt.ylabel('MAPE')
        plot.get_legend().remove()
    plt.xlabel('Occlusion factor, \u03B8')

    plt.tight_layout()
    plt.show()
    # plt.savefig(f'./data/general/degradation_plot_{y_axis_metric}.pdf')


def performance_degradation_wrapper():
    # date_list = [('Houston', '20180103'), ('Austin', '20190816'), ('Seattle', '20210420'),
    #              ('Indianapolis', '20210726'), ('Charlotte', '20211018')] # , ('San Diego', '20210104')]
    # how_performance_decreases_as_synthetic_occlusion_increases3(city=date_list[0][0], date_=date_list[0][1])
    # how_performance_decreases_as_synthetic_occlusion_increases3(city=date_list[1][0], date_=date_list[1][1])
    # how_performance_decreases_as_synthetic_occlusion_increases3(city=date_list[2][0], date_=date_list[2][1])
    # how_performance_decreases_as_synthetic_occlusion_increases3(city=date_list[3][0], date_=date_list[3][1])
    # how_performance_decreases_as_synthetic_occlusion_increases3(city=date_list[4][0], date_=date_list[4][1])
    # performance_degradation_graph(date_list)
    # vis_performance_deg_results()

    date_list = [('Houston', '20200414'), ('Austin', '20190816'), ('Oklahoma City', '20180719'),
                 ('San Diego', '20181112')]
    # date_list = [('Houston', '20190327'), ('Austin', '20210922'), ('Oklahoma City', '20180719'),
    #              ('San Diego', '20181112')]
    # date_list = [('Oklahoma City', '20180719'), ('San Diego', '20181112')]

    city_list = [entry[0] for entry in date_list]
    sampler = OcclusionSampler(city_list)
    # how_performance_decreases_as_synthetic_occlusion_increases5(city=date_list[0][0], date_=date_list[0][1], sampler=sampler)
    # how_performance_decreases_as_synthetic_occlusion_increases5(city=date_list[1][0], date_=date_list[1][1], sampler=sampler)
    # how_performance_decreases_as_synthetic_occlusion_increases5(city=date_list[2][0], date_=date_list[2][1], sampler=sampler)
    # how_performance_decreases_as_synthetic_occlusion_increases5(city=date_list[3][0], date_=date_list[3][1], sampler=sampler)
    performance_degradation_graph2(date_list, y_axis_metric='mape')
    # vis_performance_deg_results()


def vis_performance_deg_results():
    city = 'Houston'
    date_ = '20200414'
    output_dir = f'./data/{city}/analysis/occlusion_progression_{date_}_sample'
    assert p.exists(output_dir)
    files = os.listdir(output_dir)
    files = [f for f in files if 'r_' in f and 'cmap' not in f]
    # print(files)
    for f in tqdm(files):
        img = cv2.imread(p.join(output_dir, f), -1)
        save_cmap(img, p.join(output_dir, 'cmap_' + f[:-4] + '.png'), palette='inferno', vmin=285, vmax=305)


def plot_mean_trend_bt_two_dates(city, date1, date2):
    """
    :param city:
    :param date1:
    :param date2:
    :return:
    """
    sns.set(style='whitegrid', context='paper', font='Times New Roman', font_scale=1.5)
    interp_1 = Interpolator(root=f'./data/{city}', target_date=date1)
    interp_2 = Interpolator(root=f'./data/{city}', target_date=date2)
    y_mean1, y_mean2 = {}, {}
    i = 0
    for c, _ in NLCD_2019_META['lut'].items():
        c = int(c)
        temp_for_c = interp_1.target.copy()
        temp_for_c[interp_1.nlcd != c] = 0
        dp = temp_for_c[np.where(temp_for_c != 0)]
        if len(dp > 0):
            y_mean1[c] = np.average(dp)
    i = 0
    for c, _ in NLCD_2019_META['lut'].items():
        c = int(c)
        temp_for_c = interp_2.target.copy()
        temp_for_c[interp_2.nlcd != c] = 0
        dp = temp_for_c[np.where(temp_for_c != 0)]
        if len(dp > 0):
            y_mean2[c] = np.average(dp)
    palette = []
    df = pd.DataFrame({'class': [], 'delta': []})
    for c, _ in NLCD_2019_META['lut'].items():
        c = int(c)
        if c in y_mean1 and c in y_mean2:
            delta = y_mean2[c] - y_mean1[c]
            # log += [(c, delta)]
            x = NLCD_2019_META['class_names'][str(c)]
            new_df = pd.DataFrame({'class': [x], 'delta': [delta]})
            df = pd.concat([df, new_df], ignore_index=True)
            palette += ['#' + NLCD_2019_META['lut'][str(c)]]
    # df = pd.DataFrame(log, columns=['class', 'delta'])
    # print(df)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, y='class', x='delta', palette=palette)
    plt.xlabel(u'Difference in Mean Brightness Temperature (K)')
    plt.ylabel('NLCD Land Cover Classes')
    plt.title(f'Changes in Mean Brightness Temperature\nfrom {date1} to {date2} in {city}')
    plt.tight_layout()
    plt.savefig('./data/general/temporal_motivation_p1.svg', dpi=300)
    # plt.show()


def motivation_temporal():
    # sns.set(style='whitegrid', context='paper', font='Times New Roman')
    plot_mean_trend_bt_two_dates('Houston', '20181205', '20181221') # 56F, 62F
    # plot_mean_trend_bt_two_dates('Houston', '20181205', '20191106') # ok
    # plot_mean_trend_bt_two_dates('Houston', '20181221', '20170929')  # better, 68F
    # plot_mean_trend_bt_two_dates('Houston', '20210401', '20191106')  # ok
    # plot_mean_trend_bt_two_dates('Houston', '20210401', '20170929')  # used!


def motivation_temporal2(city='Houston'):
    sns.set(style='white', context='paper', font='Times New Roman', font_scale=1.5)
    df_path = f'./data/{city}/analysis/averages_by_date.csv'
    if not p.exists(df_path):
        root_ = f'./data/{city}'
        df = pd.read_csv(p.join(root_, 'metadata.csv'))
        dates = df['date'].values.tolist()
        dates = [str(d) for d in dates]
        log = []
        for d in tqdm(dates):
            interp = Interpolator(root=root_, target_date=d)
            theta = interp.add_occlusion(use_true_cloud=True)
            input_bitmask = np.array(~interp.synthetic_occlusion, dtype=np.bool_)
            input_bitmask[~interp.target_valid_mask] = False
            if np.any(input_bitmask):
                avg = np.average(interp.occluded_target[input_bitmask])
            else:
                avg = np.nan
            log += [(d, avg, theta)]
        df = pd.DataFrame(log, columns=['date', 'avg', 'theta'])
        df.to_csv(df_path, index=False)
        print('csv file saved to ', df_path)
    else:
        print('using csv file ', df_path)
        # plt.figure(figsize=(7.5, 5))
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), gridspec_kw={'height_ratios': [3, 1]})
        df = pd.read_csv(df_path)
        x_dates = [datetime.strptime(str(date_str), '%Y%m%d') for date_str in df['date']]
        # below_theta_max = [theta < 0.1 for theta in df['theta']]
        seasons = [get_season(x) for x in x_dates]
        sns.scatterplot(ax=axes[0], data=df, x=x_dates, y='avg', hue=seasons)
        axes[0].legend(loc='upper right', bbox_to_anchor=(1.21, 1.03))
        axes[0].set_ylabel('Average Brightness Temperature (K)')
        axes[0].set_xlabel('Date')
        # axes[0].legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5, 1.2))
        # plt.tight_layout()
        # plt.show()

        sns.barplot(ax=axes[1], data=df, x=x_dates, y='theta', color='gray')
        axes[1].set_ylabel('Occlusion factor')
        # axes[1].set_xlabel('Date')
        axes[1].get_xaxis().set_visible(False)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./data/general/temporal_motivation_p2.svg')


def motivation_spatial():
    sns.set(style='whitegrid', context='paper', font='Times New Roman', font_scale=1.5)
    city = 'San Antonio'
    date_ = '20190816'
    interp = Interpolator(root=f'./data/{city}', target_date=date_)
    interp.plot_violins(show=False, include_class_agnostic=True)
    # plt.show()
    plt.savefig('./data/general/motivation_spatial.pdf')


def hot_zone_wrapper():
    count_hotzones_freq_for(city='Houston', temp_type='st', threshold=315)
    count_hotzones_freq_for(city='Los Angeles', temp_type='st', threshold=320)
    count_hotzones_freq_for(city='Chicago', temp_type='st', threshold=305)


def results_figure():
    """
    This function produces the following outputs for each case study:
    * color mapped bt output
    * color mapped st output
    * color mapped input
    :return:
    """
    def _export_row(city, date_, vmin=None, vmax=None):
        palette = 'inferno'
        b10 = cv2.imread(f'./data/{city}/bt_series/LC08_B10_{date_}.tif', -1)
        bt = cv2.imread(f'./data/{city}/output_referenced/bt/bt_{date_}.tif', -1)
        st = cv2.imread(f'./data/{city}/output_referenced/st/st_{date_}.tif', -1)
        assert b10 is not None
        assert bt is not None
        assert st is not None
        # out_dir = f'./data/{city}/analysis/cmap_out_{date_}/'
        out_dir = f'./data/general/results_{city}_{date_}/'
        if not p.exists(out_dir):
            os.mkdir(out_dir)
        # copy image that do not need cmap
        shutil.copyfile(src=f'./data/{city}/TOA_RGB/RGB/LC08_RGB_{date_}.png', dst=p.join(out_dir, f'rgb_{date_}.png'))
        save_cmap(b10, p.join(out_dir, f'b10_{date_}.png'), palette=palette, vmin=vmin, vmax=vmax)
        save_cmap(bt, p.join(out_dir, f'bt_{date_}.png'), palette=palette, vmin=vmin, vmax=vmax)
        save_cmap(st, p.join(out_dir, f'st_{date_}.png'), palette=palette, vmin=vmin, vmax=vmax)
        # metadata
        interp = Interpolator(root=f'./data/{city}', target_date=date_)
        theta = interp.add_occlusion(use_true_cloud=True)
        with open(p.join(out_dir, 'readme.txt'), 'w') as f:
            f.write(f'theta = {theta:.2f}')
        # save color bar
        sns.set_theme(context='paper', font='Times New Roman')
        X, Y = np.mgrid[-2:3, -2:3]
        Z = np.random.rand(*X.shape)
        FIGSIZE = (3, 3)
        mpb = plt.pcolormesh(X, Y, Z, cmap=palette, vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        plt.colorbar(mpb, ax=ax)
        ax.remove()
        plt.savefig(p.join(out_dir, f'cbar_{date_}.pdf'))
        plt.tight_layout()
        plt.close()
        print('files saved to directory ', out_dir)
        return

    # _export_row('Houston', '20220114', vmin=280, vmax=310)
    # _export_row('New York', '20170723', vmin=280, vmax=320)
    _export_row('Jacksonville', '20171203', vmin=290, vmax=320)
    # _export_row('San Francisco', '20200606', vmin=280, vmax=320)
    # _export_row('Phoenix', '20180107', vmin=280, vmax=320)


def vis_uhie_wrt_baseline(city, hash_code=None):
    sns.set(style='white', context='paper', font='Times New Roman', font_scale=1.5)
    output_dir = f'./data/{city}/analysis/'
    if not p.exists(output_dir):
        raise FileNotFoundError()
    files = os.listdir(output_dir)
    files = [f for f in files if 'average_temp_trend' in f]
    files = [f for f in files if 'csv' in f]
    files = [f for f in files if '._' not in f]  # neglect temporary files
    if len(files) == 0:
        raise FileNotFoundError('No corresponding csv files found in ', output_dir)
    elif len(files) > 1:
        if hash_code is None:
            raise FileExistsError('Multiple csv files found. Please specify a hashcode')
        else:  # hash code is specified
            files = [f for f in files if hash_code in f]
            if len(files) == 0:
                raise FileNotFoundError('No csv file matches with the specified hash code: ', hash_code)
            elif len(files) > 1:
                raise FileExistsError('Hash collision')
    # only 1 matching csv file exists
    assert (len(files) == 1)
    yprint(f'Parsing dataframe from {files[0]}')
    df = pd.read_csv(p.join(output_dir, files[0]))
    df[df < 2] = np.nan
    x_dates = [datetime.strptime(str(date_str), '%Y%m%d') for date_str in df['date']]
    df['diff'] = df['developed'] - df['natural']
    sns.lineplot(data=df, x=x_dates, y='diff')
    plt.show()

# def vis_wetland(city='Jacksonville'):

def main():
    # read_npy_stack(path='data/Houston/output_timelapse/')
    # vis_heat(path='data/Houston/output_timelapse/')
    # calc_avg_temp_per_class_over_time(city='Jacksonville')
    # plot_avg_temp_per_class_over_time(city='Houston', hash_code='f44b')
    # count_hotzones_freq_for(city='Houston', temp_type='st', threshold=310)
    # count_hotzones_freq_for(city='Los Angeles', temp_type='st', threshold=315)
    # count_hotzones_freq_for(city='Chicago', temp_type='st', threshold=300)
    # how_performance_decreases_as_synthetic_occlusion_increases2('Austin', '20190816', added_cloud_dates=[20180728, 20200514, 20200530, 20180813, 20220520, 20211227])
    # how_performance_decreases_as_synthetic_occlusion_increases2('Seattle', '20210420', [20171205, 20180615, 20201026, 20171002, 20200604, 20170308, 20170612])
    # how_performance_decreases_as_synthetic_occlusion_increases2('Houston', '20180103', [20220319, 20190701, 20190717, 20210706, 20211010, 20210316, 20220420])
    # performance_degradation_graph()
    performance_degradation_wrapper()
    # motivation_temporal()
    # motivation_temporal2()
    # motivation_spatial()
    # hot_zone_wrapper()
    # results_figure()
    # vis_wetland()

    # vis_uhie_wrt_baseline('New York')

if __name__ == '__main__':
    main()
