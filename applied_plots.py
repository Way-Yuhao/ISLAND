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
from datetime import date, timedelta, datetime
from config import *
from interpolator import Interpolator
from util.helper import rprint, yprint, hash_, pjoin
from util.geo_reference import save_geotiff


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
    timelapse_dir = f'./data/{city}/output/'
    assert p.exists(timelapse_dir)
    output_dir = f'./data/{city}/analysis/'
    if not p.exists(output_dir):
        os.mkdir(output_dir)
    print('saving outputs to ', output_dir)
    files = [f for f in os.listdir(timelapse_dir) if '.npy' in f]
    files = [f for f in files if '_st' in f]
    files = natsorted(files)
    print(f'Got {len(files)} files, with dates ranging from {files[0][8:16]} to {files[-1][8:16]}')
    f0 = files[0]
    # print(f0)
    interp0 = Interpolator(root=f'./data/{city}/', target_date=f0[8:16])
    nlcd = interp0.nlcd
    df = pd.DataFrame()
    for f in tqdm(files, desc='Scanning predicted frames'):
        new_row = {'date': f[8:16]}
        prediction = np.load(p.join(timelapse_dir, f))
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
    df.replace(-1, np.inf)
    plt.figure(figsize=(15, 5))
    sns.set(style='whitegrid')
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
    plt.title(f'Mean brightness temperature of each land cover class for {city}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=5)
    plt.ylabel('Brightness Temperature (K)')
    plt.xlabel('Date')
    plt.tight_layout()
    # ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')
    plt.show()


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
        plt.imshow(aggregate, cmap='inferno', vmin=0, vmax=50)
        plt.colorbar(label=f'Number of day exceeding {threshold} Kelvin')
        plt.title(f'Hot zones '
                  f'from {files[0][3:11]} to {files[-1][3:11]} in {city}')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.tight_layout()

        if not p.exists(f'./data/{city}/analysis/'):
            os.mkdir(f'./data/{city}/analysis/')
        np.save(f'./data/{city}/analysis/hotzones_{threshold}k.npy', aggregate)
        # plt.savefig(f'./data/{city}/analysis/hotzones_{threshold}k.png')
        save_geotiff(city, aggregate, files[0][3:11], out_path=f'./data/{city}/analysis/hotzones_{threshold}k.tif')
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


def performance_degradation_graph():
    log_path = './data/general/performance_degradation.csv'
    df = pd.read_csv(log_path)
    print(df)
    sns.set_theme()
    sns.set_context("paper")
    plot = sns.lineplot(data=df, y='mae', x='syn_occlusion_perc', hue='city')
    plt.show()


def plot_mean_trend_bt_two_dates(city, date1, date2):
    """
    :param city:
    :param date1:
    :param date2:
    :return:
    """
    sns.set(style='whitegrid', context='paper', font='Times New Roman')
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
    sns.barplot(data=df, y='class', x='delta', palette=palette)
    plt.xlabel(u'Difference in Mean Brightness Temperature (K)')
    plt.ylabel('NLCD Land Cover Classes')
    plt.title(f'Changes in Mean Brightness Temperature\nfrom {date1} to {date2} in {city}')
    plt.tight_layout()
    plt.show()


def motivation_temporal():
    # sns.set(style='whitegrid', context='paper', font='Times New Roman')
    # plot_mean_trend_bt_two_dates('Houston', '20181205', '20181221') # 56F, 62F
    # plot_mean_trend_bt_two_dates('Houston', '20181205', '20191106') # ok
    # plot_mean_trend_bt_two_dates('Houston', '20181221', '20170929')  # better, 68F
    # plot_mean_trend_bt_two_dates('Houston', '20210401', '20191106')  # ok
    plot_mean_trend_bt_two_dates('Houston', '20210401', '20210924')  # used!

def hot_zone_wrapper():
    count_hotzones_freq_for(city='Houston', temp_type='st', threshold=315)
    # count_hotzones_freq_for(city='Los Angeles', temp_type='st', threshold=320)
    # count_hotzones_freq_for(city='Chicago', temp_type='st', threshold=305)

def main():
    # read_npy_stack(path='data/Houston/output_timelapse/')
    # vis_heat(path='data/Houston/output_timelapse/')
    # calc_avg_temp_per_class_over_time(city='Chicago')
    # plot_avg_temp_per_class_over_time(city='Chicago')
    # count_hotzones_freq_for(city='Houston', temp_type='st', threshold=310)
    # count_hotzones_freq_for(city='Los Angeles', temp_type='st', threshold=310)
    # count_hotzones_freq_for(city='Austin', temp_type='st', threshold=310)
    # how_performance_decreases_as_synthetic_occlusion_increases2('Austin', '20190816', added_cloud_dates=[20180728, 20200514, 20200530, 20180813, 20220520, 20211227])
    # how_performance_decreases_as_synthetic_occlusion_increases2('Seattle', '20210420', [20171205, 20180615, 20201026, 20171002, 20200604, 20170308, 20170612])
    # how_performance_decreases_as_synthetic_occlusion_increases2('Houston', '20180103', [20220319, 20190701, 20190717, 20210706, 20211010, 20210316, 20220420])
    # performance_degradation_graph()
    # motivation_temporal()
    hot_zone_wrapper()




if __name__ == '__main__':
    main()
