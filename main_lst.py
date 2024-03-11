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
from rich.progress import Progress
import multiprocessing as mp
from interpolators.lst_interpolator import LST_Interpolator as Interpolator
# from interpolators.bt_interpolator import BT_Interpolator as Interpolator
# from batch_eval import solve_all_bt, move_bt, compute_st_for_all
from util.helper import get_season, rprint, yprint, timer, monitor, alert, deprecated, capture_stdout


@capture_stdout
def solve_one_lst(data_dir: str, date_: str, resume: bool = False):
    if resume:
        existing_output_files = os.listdir(p.join(data_dir, 'output'))
        current_date_files = [f for f in existing_output_files if date_ in f and '_st.npy' in f]
        if len(current_date_files) > 0:
            print(f'Found outputs for date {date_}. Skipped.')
            return
    yprint(f'Evaluating {date_}')
    interp = Interpolator(root=data_dir, target_date=date_)
    interp.add_occlusion(use_true_cloud=True)
    interp.run_interpolation(spatial_kern_size=75)  # saves results to output
    return


def solve_all_lst_parallel(data_dir: str, resume: bool = False):
    num_cores = mp.cpu_count() - 2
    yprint('Computing LST in parallel using {} cores.'.format(num_cores))
    df = pd.read_csv(p.join(data_dir, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = natsorted([str(d) for d in dates])
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Running ISLAND...", total=len(dates))
        with mp.Pool(num_cores) as pool:
            # pool.starmap(solve_one_lst, [(data_dir, d, resume) for d in dates])
            results = [pool.apply_async(solve_one_lst, args=(data_dir, d, resume),
                                        callback=lambda _: progress.update(task_id, advance=1)) for d in dates]
            # Wait for all tasks to complete
            for result in results:
                result.get()
    return


def solve_all_lst(data_dir: str, resume: bool = False):
    """
    Generates timelapse for all available input frames without adding synthetic occlusion
    :param data_dir:
    :param resume: skip existing outputs and resume at the next frame
    :return:
    """
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


def geo_reference_lst(data_dir: str, mode: str = 'full', output_dir: str = 'output_referenced'):
    """
    Geo-reference the outputs of the interpolation for the entire city.
    :param data_dir:
    :return:
    """
    output_dir = p.join(data_dir, output_dir, 'lst')
    if not p.exists(output_dir):
        os.mkdir(output_dir)
    # geo-reference land surface temperature
    st_dir = p.join(data_dir, 'output')
    st_files = os.listdir(st_dir)
    if mode == 'full':
        st_files = [f for f in st_files if '_st.npy' in f]
    elif mode == 'temporal':
        st_files = [f for f in st_files if '_temporal.npy' in f]
    elif mode == 'spatial':
        st_files = [f for f in st_files if '_spatial.npy' in f]
    else:
        raise ValueError(f'Invalid mode {mode}.')
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
    print(f'Loading from {args.dir}')
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
    if not args.skip_to_ref:
        # solve_all_lst(data_dir=args.dir, resume=False)
        solve_all_lst_parallel(data_dir=args.dir, resume=False)
    # geo_reference_lst(data_dir=args.dir)
    geo_reference_lst(data_dir=args.dir, mode='full', output_dir='output_referenced')
    if not p.exists(p.join(args.dir, 'output_referenced_temporal')):
        os.mkdir(p.join(args.dir, 'output_referenced_temporal'))
    geo_reference_lst(data_dir=args.dir, mode='temporal', output_dir='output_referenced_temporal')
    if not p.exists(p.join(args.dir, 'output_referenced_spatial')):
        os.mkdir(p.join(args.dir, 'output_referenced_spatial'))
    geo_reference_lst(data_dir=args.dir, mode='spatial', output_dir='output_referenced_spatial')
    alert(f'Interpolation for {base_dir} finished.')


def main():
    process_city_lst()


if __name__ == '__main__':
    main()
