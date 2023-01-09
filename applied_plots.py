"""
Generates plots to be used in the paper
"""
import os
import os.path as p
from natsort import natsorted
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from tqdm import tqdm
from config import *
from interpolator import Interpolator
from util.helper import rprint, yprint, hash_


def plot_avg_temp_per_class_over_time(city=" "):
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
    timelapse_dir = f'./data/{city}/output_timelapse/'
    assert p.exists(timelapse_dir)
    output_dir = f'./data/{city}/analysis/'
    if not p.exists(output_dir):
        os.mkdir(output_dir)
    print('saving outputs to ', output_dir)
    files = [f for f in os.listdir(timelapse_dir) if '.npy' in f]
    files = [f for f in files if '_st' in f]
    files = natsorted(files)
    print(f'Got {len(files)} files, with dates ranging from {files[0][9:17]} to {files[-1][9:17]}')
    f0 = files[0]
    # print(f0)
    interp0 = Interpolator(root=f'./data/{city}/', target_date=f0[9:17])
    nlcd = interp0.nlcd
    df = pd.DataFrame()
    for f in tqdm(files, desc='Scanning predicted frames'):
        new_row = {'date': f[9:17]}
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


def main():
    plot_avg_temp_per_class_over_time(city='Houston')


if __name__ == '__main__':
    main()
