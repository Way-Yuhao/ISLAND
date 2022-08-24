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
from util.helper import get_season


def evaluate():
    start_time = time.monotonic()
    stats_fpath = './data/spatial_intperp_series.csv'
    interp = Interpolator(root='./data/export/', target_date='20181221')
    dataset_path = os.listdir(p.join(interp.root, 'cloud'))
    print(f"Evaluating {len(dataset_path)} scenes")
    cloud_percs, maes, mses = [], [], []
    for f in dataset_path:
        if f[-3:] != 'tif' or f[:4] == 'nlcd':
            continue
        cloud_perc = interp.add_occlusion(p.join(interp.root, 'cloud', f))

        try:
            interp.spatial_interp(f=100)
            # interp.fill_average()
        except ValueError:
            pass
        mae = interp.calc_loss(print_=False, metric='mae')
        mse = interp.calc_loss(print_=False, metric='mse')
        print(f"{cloud_perc:.3%} | mae = {mae:.3f} | mse = {mse:.3f}")
        cloud_percs.append(cloud_perc)
        interp.save_output()
        maes.append(mae)
        mses.append(mse)

    print('---------------------------------')
    d = {'cloud_perc': cloud_percs, 'MAE': maes, 'MSE': mses}
    df = pd.DataFrame(data=d)
    df.to_csv(stats_fpath)
    print("CSV file saved to ", stats_fpath)
    stop_time = time.monotonic()
    print('Processing time = ', datetime.timedelta(seconds=stop_time - start_time))


def evaluate_multiprocess(num_procs=4):
    start_time = time.monotonic()
    stats_fpath = './data/spatial_intperp_series_mp.csv'
    interp = Interpolator(root='./data/export/', target_date='20181221')
    dataset_path = os.listdir(p.join(interp.root, 'cloud'))
    for f in dataset_path:  # clean up irrelevant input files
        if f[-3:] != 'tif' or f[:4] == 'nlcd':
            dataset_path.remove(f)
    print(f"Evaluating {len(dataset_path)} scenes")

    pool = Pool(num_procs)
    manager = Manager()
    lock = manager.Lock()
    r = None  # return values
    # empty lists, compatible with multi-processes
    cloud_percs, maes, mses = manager.list(), manager.list(), manager.list()
    for f in dataset_path:
        relative_path = p.join(interp.root, 'cloud', f)
        r = pool.apply_async(eval_single, args=(relative_path, lock, cloud_percs, maes, mses))
    r.get()
    pool.close()
    pool.join()
    print('---------------------------------')
    print(f"{bcolors.WARNING}May have encountered error. Scroll up to view.")
    d = {'cloud_perc': list(cloud_percs), 'MAE': list(maes), 'MSE': list(mses)}
    df = pd.DataFrame(data=d)
    df.to_csv(stats_fpath)
    print("CSV file saved to ", stats_fpath)
    stop_time = time.monotonic()
    print('Processing time = ', datetime.timedelta(seconds=stop_time - start_time))


def eval_single(occlusion_fpath, lock, cloud_percs, maes, mses):
    interp = Interpolator(root='./data/export/', target_date='20181221')
    cloud_perc = interp.add_occlusion(occlusion_fpath)
    mae, mse = None, None
    try:
        # interp.fill_average()
        interp.spatial_interp(f=75)
        mae = interp.calc_loss(print_=False, metric='mae')
        mse = interp.calc_loss(print_=False, metric='mse')
        print(f"{cloud_perc:.3%} | mae = {mae:.3f} | mse = {mse:.3f}")
        interp.save_output()
    except ValueError as e:
        print(f"{bcolors.FAIL}ERROR: {e}{bcolors.ENDC}")

    with lock:  # to ensure atomic IO operations
        cloud_percs.append(cloud_perc)
        maes.append(mae)
        mses.append(mse)
    del interp
    return


def generate_dps(city_name, n=200):
    files = os.listdir(f"./data/{city_name}/cloud/")
    fpath = f"./data/{city_name}/rand_dates.csv"
    files = [f for f in files if 'tif' in f and 'nlcd' not in f]
    dates = natsorted([f[-12:-4] for f in files])
    d1, d2, d3 = [], [], []
    for i in range(n):
        d1 += [random.choice(dates)]
        d2 += [random.choice(dates)]
        d3 += [random.choice(dates)]
        assert d1 != d2 and d2 != d3 and d3 != d1
    df = pd.DataFrame({'date_1': d1, 'date_2': d2, 'date_3': d3})
    df.to_csv(fpath, index=False)
    print('csv file saved to ', fpath)


def temp_pairwise_eval(city_name):
    """
    Pick two cloud-free frames as target and reference, then apply different synthetic clouds for both
    :param city_name:
    :return:
    """

    entries = pd.read_csv(f"./data/{city_name}/rand_dates.csv")
    target_date = '20181221'
    ref_date = '20181205'
    root_ = f'./data/{city_name}/'
    log_fpath = f"./data/{city_name}/temporal_pairwise.csv"
    assert entries is not None
    log = []
    pbar = tqdm(total=len(entries))
    for _, row in entries.iterrows():
        target_syn_cloud_date, ref_syn_cloud_date = row['date_1'], row['date_2']
        interp = Interpolator(root=root_, target_date=target_date)
        ref_syn_cloud_path = p.join(root_, 'cloud', f'LC08_cloud_houston_{target_syn_cloud_date}.tif')
        target_perc = interp.add_occlusion(ref_syn_cloud_path)
        ref_perc = interp.temporal_interp_cloud(ref_frame_date=ref_date, ref_syn_cloud_date=ref_syn_cloud_date)
        mae_loss = interp.calc_loss(print_=True, metric='mae', entire_canvas=True)
        mse_loss = interp.calc_loss(print_=True, metric='mse', entire_canvas=True)
        log += [(target_date, ref_date, target_syn_cloud_date, target_perc, ref_syn_cloud_date, ref_perc,
                 mae_loss, mse_loss)]
        interp.save_output()
        del interp
        pbar.update()
    pbar.close()
    df = pd.DataFrame(log, columns=['target_date', 'ref_date', 'target_syn_cloud_date',
                                    'target_synthetic_occlusion_percentage', 'ref_syn_cloud_date',
                                    'reference_synthetic_occlusion_percentage', 'MAE', 'MSE'])
    df.to_csv(log_fpath, index=False)
    print('csv file saved to ', log_fpath)


def plot_temporal_pairwise():
    max_clip = 1.0  # where to clip for vis
    log_fpath = "./data/temporal_pairwise.csv"
    df = pd.read_csv(log_fpath)
    df['MAE'] = np.where(df['MAE'] > max_clip, max_clip, df['MAE'])
    x = df['reference_synthetic_occlusion_percentage']
    y = df['target_synthetic_occlusion_percentage']
    z = df['MAE']

    plt.figure(num=1, figsize=(8, 5))
    g = sns.jointplot(x=x, y=y, c=z, joint_kws={"color": None, 'cmap': 'cool'})
    g.fig.colorbar(g.ax_joint.collections[0], ax=[g.ax_joint, g.ax_marg_y, g.ax_marg_x], use_gridspec=True,
                   orientation='horizontal', label=f'MAE loss, clipped at {max_clip}')
    g.set_axis_labels('occlusion percentage of the reference frame', 'occlusion percentage of the target frame', )
    plt.show()


def temp_pairwise_cycle_eval(city_name):
    """
    fixed cloud-free target, fixed synthetic occlusion, using a set of real reference frames (potentially cloudy)
    :param city_name:
    :return:
    """
    # entries = pd.read_csv(f"./data/{city_name}/rand_dates.csv")
    target_date = '20220611'
    # ref_date = '20181205'
    # FIXME: no synthetic cloud?
    root_ = f'./data/{city_name}/'
    log_fpath = f"./data/{city_name}/temporal_references.csv"
    assert not p.exists(log_fpath)
    log = []
    frames = os.listdir(p.join(root_, 'bt_series'))
    frames = [f for f in frames if 'tif' in f]
    pbar = tqdm(total=len(frames), desc='Analyzing multiple references')
    for ref_frame in frames:
        interp = Interpolator(root=root_, target_date=target_date)
        interp.occluded_target = interp.target  # assume no cloud

        ref_date = ref_frame[9:17]
        ref_perc = interp.temporal_interp(ref_frame_date=ref_date)

        mae_loss = interp.calc_loss(print_=True, metric='mae', entire_canvas=True)
        mse_loss = interp.calc_loss(print_=True, metric='mse', entire_canvas=True)
        log += [(target_date, ref_date, np.nan, np.nan, ref_perc, mae_loss, mse_loss)]
        interp.save_output()
        del interp
        pbar.update()

        # target_syn_cloud_date, ref_syn_cloud_date = row['date_1'], row['date_2']
        # interp = Interpolator(root=root_, target_date=target_date)
        # ref_syn_cloud_path = p.join(root_, 'cloud', f'LC08_cloud_houston_{target_syn_cloud_date}.tif')
        # target_perc = interp.add_occlusion(ref_syn_cloud_path)
        # ref_perc = interp.temporal_interp_cloud(ref_frame_date=ref_date, ref_syn_cloud_date=ref_syn_cloud_date)
        # mae_loss = interp.calc_loss(print_=True, metric='mae', entire_canvas=True)
        # mse_loss = interp.calc_loss(print_=True, metric='mse', entire_canvas=True)
        # log += [(target_date, ref_date, target_syn_cloud_date, target_perc, ref_syn_cloud_date, ref_perc,
        #          mae_loss, mse_loss)]
        # interp.save_output()
        # del interp
        # pbar.update()
    pbar.close()
    df = pd.DataFrame(log, columns=['target_date', 'ref_date', 'target_syn_cloud_date',
                                    'target_synthetic_occlusion_percentage',
                                    'reference_gt_occlusion_percentage', 'MAE', 'MSE'])
    df.to_csv(log_fpath, index=False)
    print('csv file saved to ', log_fpath)


def plot_temporal_cycle(city_name):
    max_clip = 10  # where to clip for vis
    log_fpath = f"./data/{city_name}/temporal_references.csv"
    df = pd.read_csv(log_fpath)
    df['MAE'] = np.where(df['MAE'] > max_clip, max_clip, df['MAE'])
    ref_dates = df['ref_date']
    ref_dates = [datetime.datetime.strptime(str(date_str), '%Y%m%d') for date_str in ref_dates]
    seasons = [get_season(x) for x in ref_dates]
    # y = df['target_synthetic_occlusion_percentage']
    y = df['MAE']
    sns.set_style('darkgrid')
    plt.figure(num=1, figsize=(8, 5))
    # g = sns.jointplot(x=x, y=y, c=z, joint_kws={"color": None, 'cmap': 'cool'})
    # g.fig.colorbar(g.ax_joint.collections[0], ax=[g.ax_joint, g.ax_marg_y, g.ax_marg_x], use_gridspec=True,
    #                orientation='horizontal', label=f'MAE loss, clipped at {max_clip}')
    # g.set_axis_labels('occlusion percentage of the reference frame', 'occlusion percentage of the target frame', )
    sns.scatterplot(x=ref_dates, y=y, hue=seasons)
    sns.lineplot(x=ref_dates, y=df['reference_gt_occlusion_percentage'], color='black')
    plt.show()


def main():
    pass


if __name__ == '__main__':
    # main()
    # evaluate()
    # evaluate_multiprocess(num_procs=10)
    # generate_dps(city_name='Phoenix')
    # temp_pairwise_eval(city_name='Phoenix')
    # plot_temporal_pairwise()
    temp_pairwise_cycle_eval(city_name='Phoenix')
    # plot_temporal_cycle(city_name='Phoenix')
