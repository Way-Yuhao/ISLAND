import os
import os.path as p
import numpy as np
import time
import argparse
import shutil
import pandas as pd
import json
from tqdm import tqdm
from natsort import natsorted
from interpolators.lst_interpolator import LST_Interpolator as Interpolator
# from interpolators.bt_interpolator import BT_Interpolator as Interpolator
from batch_eval import solve_all_bt, move_bt, compute_st_for_all
from util.helper import get_season, rprint, yprint, timer, monitor, alert, deprecated
from util.geo_reference import geo_ref_copy


def solve_all_lst(city_name, resume=False):
    """
    Generates timelapse for all available input frames without adding synthetic occlusion
    :param city_name:
    :param resume: skip existing outputs and resume at the next frame
    :return:
    """
    root_ = f'./data/{city_name}/'
    df = pd.read_csv(p.join(root_, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = natsorted([str(d) for d in dates])
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
        interp.run_interpolation()  # saves results to output

        # geo reference





@monitor
def process_city_lst():
    """
    Computes brightness temperature and surface temperature for a given city. Require inputs to be downloaded
    in advance.
    :return:
    """
    yprint('Deprecated function. This method runs interpolation on bt and then compute lst.')
    parser = argparse.ArgumentParser(description='Process specify city name.')
    parser.add_argument('-c', nargs='+', required=True,
                        help='Process specify city name.')
    parser.add_argument('-r', required=False, action='store_true',
                        help='Toggle to resume from previous run. Will not overwrite files.')
    parser.add_argument('--skip_to_ref', required=False, action='store_true',
                        help='Toggle to skip to geo-reference step.')
    args = parser.parse_args()
    RESUME = args.r
    CITY_NAME = ""
    for entry in args.c:
        CITY_NAME += entry + " "
    CITY_NAME = CITY_NAME[:-1]

    yprint(f'-------- Processing {CITY_NAME} --------')
    if p.exists(f'./data/{CITY_NAME}/output_referenced/'):
        shutil.rmtree(f'./data/{CITY_NAME}/output_referenced/')
        yprint(f'Removing ./data/{CITY_NAME}/output_referenced/')
    if not args.skip_to_ref:
        if RESUME:
            yprint('WARNING: resume mode is on')
            if p.exists(f'./data/{CITY_NAME}/output_bt'):
                shutil.rmtree(f'./data/{CITY_NAME}/output_bt')
                yprint(f'Removing ./data/{CITY_NAME}/output_bt')
            if p.exists(f'./data/{CITY_NAME}/output_st'):
                shutil.rmtree(f'./data/{CITY_NAME}/output_st')
                yprint(f'Removing ./data/{CITY_NAME}/output_st')
            time.sleep(2)  # allow previous messages to print
        elif p.exists(f'./data/{CITY_NAME}/output'):
            raise FileExistsError(f'Output directory ./data/{CITY_NAME}/output/ already exists. '
                                  f'Please ether turn \'resume\' on or remove the existing '
                                  f'directory.')
        solve_all_lst(city_name=CITY_NAME, resume=RESUME)
    geo_reference_outputs(CITY_NAME)
    alert(f'Interpolation for {CITY_NAME} finished.')


def main():
    process_city_lst()


if __name__ == '__main__':
    main()
