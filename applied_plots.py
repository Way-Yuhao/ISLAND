"""
Generates plots to be used in the paper
"""
import os
import os.path as p
from natsort import natsorted
import numpy as np
import seaborn as sns
import wandb
from tqdm import tqdm
from util.helper import rprint, yprint


def plot_avg_temp_per_class_over_time(city=" "):
    """
    Generates a line graph of average temperature of a given landcover class over time,
    for a particular city.
    :param city:
    :return:
    """
    # scanning files
    yprint(f'Building temperature trend per class for {city}...')
    output_dir = f'./data/{city}/output_timelapse/'
    assert p.exists(output_dir)
    files = [f for f in os.listdir(output_dir) if '.npy' in f]
    files = [f for f in files if '_st' in f]
    files = natsorted(files)
    print(f'Got {len(files)} files, with dates ranging from {files[0][9:17]} to {files[-1][9:17]}')


def main():
    plot_avg_temp_per_class_over_time(city='Houston')


if __name__ == '__main__':
    main()
