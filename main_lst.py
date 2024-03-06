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
# from batch_eval import solve_all_bt, move_bt, compute_st_for_all
from util.helper import get_season, rprint, yprint, timer, monitor, alert, deprecated


def solve_all_lst(data_dir: str, resume: bool = False):
    """
    Generates timelapse for all available input frames without adding synthetic occlusion
    :param data_dir:
    :param resume: skip existing outputs and resume at the next frame
    :return:
    """
    # root_ = f'./data/{city_name}/'
    df = pd.read_csv(p.join(data_dir, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = natsorted([str(d) for d in dates])
    for d in tqdm(dates):
        if resume:
            existing_output_files = os.listdir(p.join(data_dir, 'output'))
            current_date_files = [f for f in existing_output_files if d in f]
            if len(current_date_files) > 0:
                print(f'Found outputs for date {d}. Skipped.')
                continue
        yprint(f'Evaluating {d}')
        interp = Interpolator(root=data_dir, target_date=d)
        interp.add_occlusion(use_true_cloud=True)
        interp.run_interpolation(spatial_kern_size=75)  # saves results to output

        # geo reference


def geo_reference_lst(data_dir: str):
    """
    Geo-reference the outputs of the interpolation for the entire city.
    :param data_dir:
    :return:
    """
    # output_dir = f'./data/{city_name}/output_referenced/lst'
    output_dir = p.join(data_dir, 'output_referenced', 'lst')
    if not p.exists(output_dir):
        os.mkdir(output_dir)
    # geo-reference land surface temperature
    # st_dir = f'./data/{city_name}/output/'
    st_dir = p.join(data_dir, 'output')
    st_files = os.listdir(st_dir)
    st_files = [f for f in st_files if '_st.npy' in f]
    for f in st_files:
        geo_ref_copy_lst(data_dir, f, p.join(output_dir, f'lst_{f[8:16]}.tif'))
    print(f'Geo-reference finished for {os.path.basename(data_dir)}.')


def geo_ref_copy_lst(data_dir, npy_filename, out_path='default'):
    """
    Copies geo-reference data from corresponding GeoTIFF input files and past to output
    :param data_dir:
    :param npy_filename: input filename. Requires file to be stored in .../{city}/output_XX_/npy/ with a format
    of XX_YYYYMMDD.npy, where XX is mode and Y M D are year, month, date, respectively.
    :param out_path:
    :return:
    """
    # root_ = f'./data/{city}'
    mode = 'lst'
    date_ = npy_filename[8:16]
    npy_path = p.join(data_dir, f'output', npy_filename)
    assert p.exists(npy_path)
    npy_content = np.load(npy_path)
    assert npy_content is not None
    reference_geotiff_path = p.join(data_dir, f'lst/LC08_ST_B10_{date_}.tif')
    assert p.exists(reference_geotiff_path), \
        f'Reference file {reference_geotiff_path} does not exist'
    ref_img = rasterio.open(reference_geotiff_path)
    if out_path == 'default':
        p.join(data_dir, f'output_referenced/{mode}/{mode}_{date_}.tif')
    out_tif = rasterio.open(out_path, 'w',
                            driver='Gtiff', height=ref_img.height, width=ref_img.width,
                            count=1, crs=ref_img.crs, transform=ref_img.transform,
                            dtype=npy_content.dtype)
    out_tif.write(npy_content, 1)
    ref_img.close()
    out_tif.close()


# @monitor
@timer
@ deprecated # use process_city_lst instead
def process_city_lst_old():
    """
    Computes brightness temperature and surface temperature for a given city. Require inputs to be downloaded
    in advance.
    :return:
    """
    parser = argparse.ArgumentParser(description='Process specify city name.')
    # parser.add_argument('-c', nargs='+', required=True,
    #                     help='Process specify city name.')
    # parser.add_argument('-r', required=False, action='store_true',
    #                     help='Toggle to resume from previous run. Will not overwrite files.')
    parser.add_argument('--remove_output', required=False, action='store_true',
                        help='Toggle to remove existing output directory.')
    parser.add_argument('--remove_output_ref', required=False, action='store_true',
                        help='Toggle to remove existing output_referenced directory.')
    parser.add_argument('--skip_to_ref', required=False, action='store_true',
                        help='Toggle to skip to geo-reference step.')
    args = parser.parse_args()
    CITY_NAME = ""
    for entry in args.c:
        CITY_NAME += entry + " "
    CITY_NAME = CITY_NAME[:-1]

    yprint(f'-------- Processing {CITY_NAME} --------')
    # remove existing output directories when prompted
    if args.remove_output:
        shutil.rmtree(f'./data/{CITY_NAME}/output')
        yprint(f'Removed output directory for {CITY_NAME}')
    if args.remove_output_ref:
        shutil.rmtree(f'./data/{CITY_NAME}/output_referenced')
        yprint(f'Removed output_referenced directory for {CITY_NAME}')
    # check if output directories already exist
    if p.exists(f'./data/{CITY_NAME}/output'):
        alert(f'Output directory ./data/{CITY_NAME}/output/ already exists. May overwrite existing files.')
    else:
        os.mkdir(f'./data/{CITY_NAME}/output')
    if p.exists(f'./data/{CITY_NAME}/output_referenced'):
        alert(f'Output_referenced directory ./data/{CITY_NAME}/output_referenced/ already exists. '
              f'May overwrite existing files.')
    else:
        os.mkdir(f'./data/{CITY_NAME}/output_referenced')
    # run interpolation
    if not args.skip_to_ref:
        solve_all_lst(city_name=CITY_NAME, resume=False)
    geo_reference_lst(CITY_NAME)
    alert(f'Interpolation for {CITY_NAME} finished.')


@monitor
@timer
def process_city_lst():
    """
    Computes brightness temperature and surface temperature for a given region.
    Require inputs to be downloaded
    in advance.
    :return:
    """
    parser = argparse.ArgumentParser(description='Process specify city name.')
    parser.add_argument('--dir', required=True, help='Specify directory for data .')
    parser.add_argument('--remove_output', required=False, action='store_true',
                        help='Toggle to remove existing output directory.')
    parser.add_argument('--remove_output_ref', required=False, action='store_true',
                        help='Toggle to remove existing output_referenced directory.')
    parser.add_argument('--skip_to_ref', required=False, action='store_true',
                        help='Toggle to skip to geo-reference step.')
    args = parser.parse_args()
    assert p.exists(args.dir), f'Directory {args.dir} does not exist.'
    base_dir = os.path.basename(args.dir.rstrip('/'))
    yprint(f'-------- Processing {base_dir} --------')
    print(f'Loading from {base_dir}')
    # remove existing output directories when prompted
    if args.remove_output:
        shutil.rmtree(p.join(args.dir, 'output'))
        yprint(f'Removed output directory for {base_dir}')
    if args.remove_output_ref:
        shutil.rmtree(p.join(args.dir, 'output_referenced'))
        yprint(f'Removed output_referenced directory for {base_dir}')
    # check if output directories already exist
    if p.exists(p.join(args.dir, 'output')):
        alert(f'Output directory {p.join(args.dir, "output")} already exists. May overwrite existing files.')
    else:
        os.mkdir(p.join(args.dir, 'output'))
    if p.exists(p.join(args.dir, 'output_referenced')):
        alert(f'Output_referenced directory {p.join(args.dir, "output_referenced")} already exists. '
              f'May overwrite existing files.')
    else:
        os.mkdir(p.join(args.dir, 'output_referenced'))
    # run interpolation
    solve_all_lst(data_dir=args.dir, resume=False)
    geo_reference_lst(data_dir=args.dir)
    alert(f'Interpolation for {base_dir} finished.')


def main():
    process_city_lst()


if __name__ == '__main__':
    main()
