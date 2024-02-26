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
import rasterio
from interpolators.lst_interpolator import LST_Interpolator as Interpolator
# from interpolators.bt_interpolator import BT_Interpolator as Interpolator
from batch_eval import solve_all_bt, move_bt, compute_st_for_all
from util.helper import get_season, rprint, yprint, timer, monitor, alert, deprecated


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


def geo_reference_lst(city_name):
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
    if not p.exists(output_dir):
        os.mkdir(output_dir)
    os.mkdir(p.join(output_dir, 'lst'))
    # land surface temperature
    st_dir = f'./data/{city_name}/output/'
    st_files = os.listdir(st_dir)
    st_files = [f for f in st_files if '_st.npy' in f]
    for f in st_files:
        geo_ref_copy_lst(city_name, f, p.join(output_dir, 'lst', f[8:16] + '.tif'))
        # geo_ref(bounding_box, p.join(st_dir, f), p.join(output_dir, 'st', f[:-4] + '.tif'))
    print(f'Geo-reference finished for {city_name}.')


def geo_ref_copy_lst(city, npy_filename, out_path='default'):
    """
    Copies geo-reference data from corresponding GeoTIFF input files and past to output
    :param city:
    :param npy_filename: input filename. Requires file to be stored in .../{city}/output_XX_/npy/ with a format
    of XX_YYYYMMDD.npy, where XX is mode and Y M D are year, month, date, respectively.
    :param out_path:
    :return:
    """
    root_ = f'./data/{city}'
    mode = 'lst'
    date_ = npy_filename[8:16]
    npy_path = p.join(root_, f'output', npy_filename)
    assert p.exists(npy_path)
    npy_content = np.load(npy_path)
    assert npy_content is not None
    reference_geotiff_path = p.join(root_, f'lst/LC08_ST_B10_{date_}.tif')
    assert p.exists(reference_geotiff_path), \
        f'Reference file {reference_geotiff_path} does not exist'
    ref_img = rasterio.open(reference_geotiff_path)
    if out_path == 'default':
        p.join(root_, f'output_referenced/{mode}/{mode}_{date_}.tif')
    out_tif = rasterio.open(out_path, 'w',
                            driver='Gtiff', height=ref_img.height, width=ref_img.width,
                            count=1, crs=ref_img.crs, transform=ref_img.transform,
                            dtype=npy_content.dtype)
    out_tif.write(npy_content, 1)
    ref_img.close()
    out_tif.close()

# @monitor
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
    geo_reference_lst(CITY_NAME)
    alert(f'Interpolation for {CITY_NAME} finished.')


def main():
    process_city_lst()


if __name__ == '__main__':
    main()
