import argparse
import time
import datetime
import os
import os.path as p
from multiprocessing import Manager, Pool
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
from interpolators.bt_interpolator import BT_Interpolator as Interpolator
from natsort import natsorted
import random
import wandb
import shutil
from tqdm import tqdm
from rich.progress import track
from config import *
from util.helper import get_season, rprint, yprint, timer, hash_
from region_sampler import add_missing_image


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


def temp_pairwise_cycle_eval_single(root_, target_date, ref_frame, log):
    interp = Interpolator(root=root_, target_date=target_date)
    interp.occluded_target = interp.target  # assume no cloud

    ref_date = ref_frame[9:17]
    ref_perc = interp.temporal_interp(ref_frame_date=ref_date)

    mae_loss = interp.calc_loss(print_=True, metric='mae', entire_canvas=True)
    mse_loss = interp.calc_loss(print_=True, metric='mse', entire_canvas=True)
    log += [(target_date, ref_date, np.nan, np.nan, ref_perc, mae_loss, mse_loss)]
    interp.save_output()
    del interp


@timer
def temp_pairwise_cycle_eval_mp(city_name):
    """
    fixed cloud-free target, fixed synthetic occlusion, using a set of real reference frames (potentially cloudy)
    :param city_name:
    :return:
    """
    target_date = '20220102'
    # vars for multiprocess
    num_procs = 10
    pool = Pool(num_procs)
    manager = Manager()
    lock = manager.Lock()
    r = None  # return values
    # FIXME: no synthetic cloud?
    root_ = f'./data/{city_name}/'
    log_fpath = f"./data/{city_name}/temporal_references.csv"
    assert not p.exists(log_fpath)
    log = manager.list()
    frames = os.listdir(p.join(root_, 'bt_series'))
    frames = [f for f in frames if 'tif' in f]
    for ref_frame in frames:
        r = pool.apply_async(temp_pairwise_cycle_eval_single, args=(root_, target_date, ref_frame, log))
    r.get()
    pool.close()
    pool.join()
    print('---------------------------------')
    yprint('Multi-process pool finished. Scroll up to view potential errors')
    df = pd.DataFrame(list(log), columns=['target_date', 'ref_date', 'target_syn_cloud_date',
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
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
    # plt.figure(num=1, figsize=(8, 5))
    sns.scatterplot(ax=axes[0], x=ref_dates, y=y, hue=seasons)
    axes[0].set_title('Choice of reference frame vs. MAE loss')
    axes[0].set_xlabel('Date of reference frame')
    axes[0].set_ylabel(f'MAE loss, clipped at {max_clip}')

    sns.lineplot(ax=axes[1], x=ref_dates, y=df['reference_gt_occlusion_percentage'], color='black')
    # axes[1].set_title('Cloud coverages vs. time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cloud coverage (%)')
    plt.show()


###################### experiments with synthetic occlusions ######################

@timer
def timelapse_with_synthetic_occlusion(city_name, occlusion_size, num_occlusions, resume=False):
    """
    Generates timelapses of BT for a given city while adding random synthetic occlusion.
    Evaluates loss only on synthetically occluded areas.
    :param resume: skip existing outputs and resume at the next frame
    :param city_name:
    :return:
    """
    if not resume and p.exists(f'./data/{city_name}/output/'):
        raise FileExistsError(f'Output directory ./data/{city_name}/output/ already exists'
                              f'please either turn \'resume\' on or remove the existing '
                              f'directory.')
    root_ = f'./data/{city_name}/'
    log_fpath = f"./data/{city_name}/output/timelapse_log.csv"
    df = pd.read_csv(p.join(root_, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = [str(d) for d in dates]
    log = []
    for d in dates:
        if resume:
            existing_output_files = os.listdir(p.join(root_, 'output'))
            current_date_files = [f for f in existing_output_files if d in f]
            if len(current_date_files) > 0:
                print(f'Found outputs for date {d}. Skipped.')
                continue
        yprint(f'Evaluating {d}')
        interp = Interpolator(root=root_, target_date=d)
        added_occlusion = interp.add_random_occlusion(size=occlusion_size, num_occlusions=num_occlusions)
        # save added occlusion
        output_filename = f'syn_occlusion_{d}'
        np.save(p.join(interp.output_path, output_filename), added_occlusion)
        plt.imshow(added_occlusion)
        plt.title(f'Added synthetic occlusion on {d}')
        output_filename = f'syn_occlusion_{d}.png'
        plt.savefig(p.join(interp.output_path, output_filename))
        interp.run_interpolation()
        loss, error_map = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
        interp.save_error_frame(mask=added_occlusion, suffix='st')
        print(f'MAE loss over synthetic occluded areas = {loss:.3f}')
        log += [(d, loss, np.count_nonzero(added_occlusion))]
    df = pd.DataFrame(log, columns=['target_date', 'mae', 'synthetic occlusion percentage'])
    df.to_csv(log_fpath, index=False)
    print('csv file saved to ', log_fpath)


def calc_error_from_outputs(city_name, output_dir, mode=None):
    """
    Find error maps from output directory and computes error
    :param city_name:
    :param output_dir:
    :param mode: full, spatial, temporal, or None (for naive)
    :return:
    """
    invalid_frame = False
    root_ = f'./data/{city_name}'
    # output_dir = f'./data/{city_name}/ablation_s750_n3/output_eval_full'
    assert p.exists(output_dir)
    yprint(f'Calculating error using files found in {output_dir}')
    df = pd.read_csv(p.join(root_, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = [str(d) for d in dates]
    log = []
    if mode == 'full':
        yprint('Using full model output')
        mode = 'st'
    elif mode == 'spatial':
        yprint('Using output from spatial channel only')
    elif mode == 'temporal':
        yprint('Using output from temporal channel only')
    elif mode is None:
        mode = 'naive'
    else:
        raise NotImplementedError()
    for d in tqdm(dates, desc='Calculating error'):
        if mode != 'naive':
            output = np.load(p.join(output_dir, f'reconst_{d}_{mode}.npy'))
        else:
            output = np.load(p.join(output_dir, f'reconst_{d}.npy'))
        syn_occlusion = np.load(p.join(output_dir, f'syn_occlusion_{d}.npy'))
        interp = Interpolator(root=root_, target_date=d)
        interp.reconstructed_target = output
        # interp.synthetic_occlusion = syn_occlusion
        syn_occlusion_perc = np.count_nonzero(syn_occlusion) / (output.shape[0] * output.shape[1])
        if syn_occlusion_perc < 0.0001:
            invalid_frame = True
            mae_loss = np.nan
            rmse_loss = np.nan
            mse_loss = np.nan
        else:
            mae_loss, error_map = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=syn_occlusion)
            rmse_loss, _ = interp.calc_loss_hybrid(metric='rmse', synthetic_only_mask=syn_occlusion)
            mse_loss, _ = interp.calc_loss_hybrid(metric='mse', synthetic_only_mask=syn_occlusion)
        real_occlusion_perc = interp.add_occlusion(use_true_cloud=True)
        total_occlusion_perc = syn_occlusion_perc + real_occlusion_perc
        log += [(d, mae_loss, rmse_loss, mse_loss, syn_occlusion_perc, real_occlusion_perc, total_occlusion_perc)]
    if not p.exists(f'./data/{city_name}/analysis/'):
        os.mkdir(f'./data/{city_name}/analysis/')
    log_fpath = f'./data/{city_name}/analysis/error_{mode}_{hash_()}.csv'
    df = pd.DataFrame(log, columns=['target_date', 'mae', 'rmse', 'mse',
                                    'synthetic occlusion %', 'real occlusion %', 'total occlusion %'])
    df.to_csv(log_fpath, index=False)
    print('------------------------------------------')
    yprint(f'log file saved to {log_fpath}')
    print('Average MAE = ', df['mae'].mean())
    print('Average RMSE = ', df['rmse'].mean())
    print('Average MSE = ', df['mse'].mean())


######### experiments with real occlusion ################

def solve_all_bt(city_name, resume=False):
    """
    Generates timelapse for all available input frames without adding synthetic occlusion
    :param resume: skip existing outputs and resume at the next frame
    :param city_name:
    :return:
    """
    root_ = f'./data/{city_name}/'
    df = pd.read_csv(p.join(root_, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = [str(d) for d in dates]
    for d in tqdm(dates):
        if resume:
            existing_output_files = os.listdir(p.join(root_, 'output'))
            current_date_files = [f for f in existing_output_files if d in f]
            if len(current_date_files) > 0:
                print(f'Found outputs for date {d}. Skipped.')
                continue
        yprint(f'Evaluating {d}')
        interp = Interpolator(root=root_, target_date=d)
        interp.add_occlusion(use_true_cloud=True)
        interp.run_interpolation()


def move_bt(city_name):
    root_ = f'./data/{city_name}/'
    assert not p.exists(p.join(root_, 'output_bt')), 'Output directory already exists'
    os.mkdir((p.join(root_, 'output_bt')))
    os.mkdir((p.join(root_, 'output_bt', 'png')))
    os.mkdir((p.join(root_, 'output_bt', 'npy')))
    # TODO: create a folder for geo-referenced data
    df = pd.read_csv(p.join(root_, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = [str(d) for d in dates]
    for d in tqdm(dates, desc='Copying files'):
        # copy npy
        bt_src = p.join(root_, 'output', f'reconst_{d}_st.npy')
        bt_dst = p.join(root_, 'output_bt', 'npy', f'bt_{d}.npy')
        shutil.copyfile(bt_src, bt_dst)
        # copy png
        bt_src = p.join(root_, 'output', f'reconst_{d}_st.png')
        bt_dst = p.join(root_, 'output_bt', 'png', f'bt_{d}.png')
        shutil.copyfile(bt_src, bt_dst)


def compute_st_for_all(city_name):
    """
    Generates emissivity-corrected surface temperature frames from existing
    interpolated brightness temperature frames
    :param city_name:
    :return:
    """
    EMIS_SCALING_FACTOR = 0.0001  # scaling factor for unit conversion.
    root_ = f'./data/{city_name}/'
    assert not p.exists(p.join(root_, 'output_st')), 'Output directory already exists'
    os.mkdir((p.join(root_, 'output_st')))
    os.mkdir((p.join(root_, 'output_st', 'png')))
    os.mkdir((p.join(root_, 'output_st', 'npy')))
    df = pd.read_csv(p.join(root_, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = [str(d) for d in dates]
    failed_dates = []
    for d in tqdm(dates, desc='Computing surface temp'):
        # brightness temperature as bt, and emissivity as emis
        bt = np.load(p.join(root_, 'output', f'reconst_{d}_st.npy')).astype('float32')
        emis = cv2.imread(p.join(root_, 'emis', f'LC08_ST_EMIS_{d}.tif'), -1)
        if emis is None:
            rprint(f'emissivity file for date {d} is not found. Attempting re-download...')
            add_missing_image(city_name=city_name, date_=d)
            emis = cv2.imread(p.join(root_, 'emis', f'LC08_ST_EMIS_{d}.tif'), -1)
            if emis is None:  # retry once
                failed_dates += [d]
                continue
        emis = emis.astype('float32') * EMIS_SCALING_FACTOR
        st = bt / emis  # brightness temperature
        # save unscaled outputs
        output_filename = f'st_{d}'
        np.save(p.join(root_, 'output_st', 'npy', output_filename), st)
        # save scaled visualization
        output_vmin = 270
        output_vmax = 330
        plt.imshow(st, cmap='magma', vmax=output_vmax, vmin=output_vmin)
        plt.title(f'Reconstructed Surface Temperature on {d}')
        plt.colorbar(label='BT(Kelvin)')
        output_filename = f'st_{d}.png'
        plt.savefig(p.join(root_, 'output_st', 'png', output_filename))
        plt.close()
    if len(failed_dates) == 0:
        yprint('No issues found in surface temperature calculation.')
    else:
        rprint(
            f'The following dates does not have surface temperature outputs: \n{failed_dates}\nCheck error messages in red above.')


if __name__ == '__main__':
    pass
    # main()
    # evaluate()
    # evaluate_multiprocess(num_procs=10)
    # generate_dps(city_name='Phoenix')
    # temp_pairwise_eval(city_name='Phoenix')
    # plot_temporal_pairwise()
    # temp_pairwise_cycle_eval_mp(city_name='Phoenix')

    # plot_temporal_cycle(city_name='Phoenix')
    # timelapse_with_synthetic_occlusion(city_name='Houston')
