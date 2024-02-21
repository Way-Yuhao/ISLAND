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
# from interpolators.lst_interpolator import LST_Interpolator as Interpolator
from interpolators.bt_interpolator import BT_Interpolator as Interpolator
from batch_eval import solve_all_bt, move_bt, compute_st_for_all
from util.helper import get_season, rprint, yprint, timer, monitor, alert, deprecated
from util.geo_reference import geo_ref_copy


def geo_reference_outputs(city_name):
    """
    Geo-reference the outputs of the interpolation for the entire city.
    :param city_name:
    :return:
    """
    # acquire bounding box
    cities_list_path = "./data/us_cities.csv"
    cols = list(pd.read_csv(cities_list_path, nrows=1))
    cities_meta = pd.read_csv(cities_list_path, usecols=[i for i in cols if i != 'notes'])
    row = cities_meta.loc[cities_meta['city'] == city_name]
    if row.empty:
        raise IndexError(f'City {city_name} is not specified in {cities_list_path}')
    scene_id = str(row.iloc[0]['scene_id'])
    if len(scene_id) == 5:
        scene_id = '0' + scene_id
    bounding_box = row.iloc[0]['bounding_box']
    assert scene_id is not np.nan, f'scene_id for {city_name} is undefined'
    assert bounding_box is not np.nan, f'bounding_box for {city_name} is undefined'
    bounding_box = json.loads(bounding_box)
    # geo-reference
    output_dir = f'./data/{city_name}/output_referenced'
    assert not p.exists(output_dir)
    os.mkdir(output_dir)
    os.mkdir(p.join(output_dir, 'st'))
    os.mkdir(p.join(output_dir, 'bt'))
    # brightness temperature
    bt_dir = f'./data/{city_name}/output_bt/npy/'
    bt_files = os.listdir(bt_dir)
    bt_files = [f for f in bt_files if '.npy' if f]
    for f in bt_files:
        geo_ref_copy(city_name, f,  p.join(output_dir, 'bt', f[:-4] + '.tif'))
        # geo_ref(bounding_box, p.join(bt_dir, f), p.join(output_dir, 'bt', f[:-4] + '.tif'))
    # surface temperature
    st_dir = f'./data/{city_name}/output_st/npy/'
    st_files = os.listdir(st_dir)
    st_files = [f for f in st_files if '.npy' if f]
    for f in st_files:
        geo_ref_copy(city_name, f, p.join(output_dir, 'st', f[:-4] + '.tif'))
        # geo_ref(bounding_box, p.join(st_dir, f), p.join(output_dir, 'st', f[:-4] + '.tif'))
    print(f'Geo-reference finished for {city_name}.')


# @monitor
@deprecated
def process_city_bt():
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
        solve_all_bt(city_name=CITY_NAME, resume=RESUME)  # solve for brightness temperature
        move_bt(city_name=CITY_NAME)
        compute_st_for_all(city_name=CITY_NAME)  # solve for surface temperature
    geo_reference_outputs(CITY_NAME)
    alert(f'Interpolation for {CITY_NAME} finished.')




def main():
    process_city_bt()


if __name__ == '__main__':
    main()
