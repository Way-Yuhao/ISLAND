import time
import datetime
import os
import os.path as p
from multiprocessing import Manager, Pool
import numpy as np
import matplotlib
import pandas as pd
from interpolator import Interpolator
from config import *


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


def main():
    pass


if __name__ == '__main__':
    # main()
    # evaluate()
    evaluate_multiprocess(num_procs=10)
