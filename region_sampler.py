__author__ = 'Yuhao Liu'
"""
Sequentially sample a defined region from earth engine. 
"""
import argparse
import os
import os.path as p
# import torch
import ee
import geemap
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import date, timedelta, datetime
from multiprocessing import Manager, Pool
import shutil
import glob
from config import *
import cv2
import pandas as pd
import tifffile
import natsort
import seaborn as sns
from matplotlib import pyplot as plt
from retry import retry
import wandb
from interpolator import Interpolator
from earth_engine_loader import SeqEarthEngineLoader
from util.helper import *

GLOBAL_REFERENCE_DATE = None  # used to calculate the validity of date for LANDSAT 8, to be defined later


def ee_init(high_volume=False):
    if high_volume is True:
        # high volume API. REQUIRES RETRY MODULE
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        yprint('Initialized High Volume API to Earth Engine')
    else:
        ee.Initialize()
        yprint('Initialized standard API to Earth Engine')
    print('---------------------------------------')
    return


def acquire_reference_date(start_date, scene_id):
    """
    Find the first date after which a LANDSAT 8 image is valid for a given scene id
    :param start_date:
    :param scene_id:
    :return:
    """
    reference_date = None
    cur_date = datetime.strptime(start_date, '%Y%m%d')
    while reference_date is None:
        cur_date_str = datetime.strftime(cur_date, '%Y%m%d')
        try:
            img = ee.Image(f'LANDSAT/LC08/C02/T1_TOA/LC08_{scene_id}_{cur_date_str}')
            img.getInfo()
        except ee.EEException as e:  # image does not exist
            cur_date = cur_date + timedelta(days=1)
            continue
        # image exists, in the case of no exception
        reference_date = cur_date_str
        return reference_date


def verify_date(date_str: str, suppressOutput=False) -> bool:
    date = datetime.strptime(date_str, '%Y%m%d')
    end_date = datetime.strptime(NULLIFY_DATES_AFTER, '%Y%m%d')
    # if (date - nullified_date).days >= 0:
    #     if not suppressOutput:
    #         print(f'Invalid date encountered: {date_str}. Need to specify a date before {NULLIFY_DATES_AFTER}')
    #     return False
    # else:
    time_delta = datetime.strptime(GLOBAL_REFERENCE_DATE, '%Y%m%d') - date
    r = time_delta.days % 16
    if r == 0:
        return True
    else:
        if not suppressOutput:
            print(f'Date {date_str} is not valid for the selected region. Consider adding {r} days.')
        return False


def calc_num_cycles(start_date_str: str, end_date_str: str) -> int:
    b1 = verify_date(start_date_str)
    b2 = verify_date(end_date_str)
    if (not b1) or (not b2):
        raise AttributeError('ERROR: invalid dates encountered')
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')
    date_delta = int((end_date - start_date).days / 16)
    return date_delta


def find_next_valid_date(date_str: str, strictly_next=False) -> str:
    date_ = datetime.strptime(date_str, '%Y%m%d')
    if verify_date(date_str, suppressOutput=True):  # current date is a valid date
        if not strictly_next:
            return date_str
        else:
            r = 16
    else:  # current date is not a valid date
        time_delta = datetime.strptime(GLOBAL_REFERENCE_DATE, '%Y%m%d') - date_
        r = time_delta.days % 16
    next_valid_time = date_ + timedelta(days=r)
    next_valid_date = next_valid_time.strftime('%Y%m%d')
    assert verify_date(next_valid_date), f"ERROR: Unable to find a next valid date after {date_str}"
    return next_valid_date


def generate_cycles(start_date: str, end_date=None, num_days=None, num_cycles=None):
    param_error_msg = "only allowed to specify exactly one of end_date, num_days, or num_cycles"
    cycles = []
    if end_date is not None:
        assert num_days is None and num_cycles is None, param_error_msg
        raise NotImplementedError()
    elif num_days is not None:
        assert end_date is None and num_cycles is None, param_error_msg
        raise NotImplementedError()
    elif num_cycles is not None:
        assert end_date is None and num_days is None, param_error_msg
        assert num_cycles >= 1, "Numer of cycles have to be >= 1"
        cur_date = find_next_valid_date(start_date)
        for cycle in range(num_cycles):
            cycles.append(cur_date)
            cur_date = find_next_valid_date(cur_date, strictly_next=True)

    yprint(f'Attempting to generate {len(cycles)} cycles, with start date = {cycles[0]} and end date = {cycles[-1]}')
    return cycles


def generate_cycles_unified(start_date: str, end_date=None, num_days=None, num_cycles=None):
    raise NotImplementedError


####################################################################################

def init_dir(output_dir, overwrite):
    assert p.exists(output_dir), f'ERROR: output directory {output_dir} does not exist.'
    if not overwrite:
        assert len(os.listdir(output_dir)) == 0, f'ERROR: directory {output_dir} is not empty.'
        os.mkdir(p.join(output_dir, 'input'))
        os.mkdir(p.join(output_dir, 'label'))
    else:
        if len(os.listdir(output_dir)) != 0:
            print(f"WARNING: directory is not empty. Overwriting existing files")
    print(f'writing files to {output_dir}...')
    return


def init_dir_train_dev(output_dir):
    assert p.exists(output_dir), f'ERROR: output directory {output_dir} does not exist.'
    assert len(os.listdir(output_dir)) == 0, f'ERROR: directory {output_dir} is not empty.'
    os.mkdir(p.join(output_dir, 'train'))
    os.mkdir(p.join(output_dir, 'dev'))
    init_dir(p.join(output_dir, 'train'), overwrite=False)
    init_dir(p.join(output_dir, 'dev'), overwrite=False)
    return


def sample_region_single(input_meta, label_meta, boundary, map_, output_dir):
    """
    Sample one crop in a pre-defined region
    :param input_meta:
    :param label_meta:
    :param boundary:
    :param map_:
    :param output_dir:
    :return:
    """
    print(f"Sampling tile {input_meta['id'][-20:]}...")
    data_loader = torch.utils.data.DataLoader(
        SeqEarthEngineLoader(root="./", geemap_obj=map_, bounding_box=boundary, image_meta=input_meta,
                             label_meta=label_meta), batch_size=1, num_workers=0)
    num_samples = len(data_loader)
    loader_iter = iter(data_loader)
    print(f'Extracting {num_samples} samples')
    for i in tqdm(range(num_samples)):
        input_, label = loader_iter.next()
        np.save(p.join(output_dir, f"input/{i}.npy"), input_)
        np.save(p.join(output_dir, f"label/{i}.npy"), label)


def sample_region_series(start_date, num_cycles, input_meta, label_meta, boundary, map_, output_dir):
    global_idx = 0
    missing_tiles = []
    cycles = generate_cycles(start_date=start_date, num_cycles=num_cycles)
    for date_ in cycles:
        cur_input_meta = input_meta.copy()
        cur_input_meta['id'] = cur_input_meta['id'].format(date_)
        try:
            data_loader = torch.utils.data.DataLoader(
                SeqEarthEngineLoader(root="./", geemap_obj=map_, bounding_box=boundary, image_meta=cur_input_meta,
                                     label_meta=label_meta), batch_size=1, num_workers=0)
            num_samples = len(data_loader)
            loader_iter = iter(data_loader)
            print(f"Sampling tile {cur_input_meta['id'][-20:]}")
            for _ in tqdm(range(num_samples)):
                input_, label = loader_iter.next()
                np.save(p.join(output_dir, f"input/{global_idx}.npy"), input_)
                np.save(p.join(output_dir, f"label/{global_idx}.npy"), label)
                global_idx += 1
        except AttributeError:
            print(f"ERROR: encountered an Attribute Error when attempting to load tile {cur_input_meta['id'][-20:]}. "
                  f"Data entry may not exist.")
            missing_tiles.append(cur_input_meta['id'])

    if not missing_tiles:
        print("All tiles have been sampled. ")
    else:
        print(f'Encountered missing {len(missing_tiles)} missing tiles: {missing_tiles}')
    print(f"Total number of crops sampled = {global_idx}")
    return


def split_train_dev_local(input_dir, output_dir, validation_split=0.2, mode='sequential'):
    init_dir_train_dev(output_dir)
    dataset_size = len(glob.glob1(p.join(input_dir, 'input'), '*.npy'))
    assert len(glob.glob1(p.join(input_dir, 'label'), '*.npy')) == dataset_size
    indices = list(range(dataset_size))
    print("Using cross-validation with a {:.0%}/{:.0%} train/dev split:".format(1 - validation_split,
                                                                                validation_split))
    if mode == 'sequential':
        split = int(np.floor(validation_split * dataset_size))
        train_indices, dev_indices = indices[split:], indices[:split]
        print('mode = sequential sampling')
        print("dev set: entry {} to {} | train set: entry {} to {}"
              .format(dev_indices[0], dev_indices[-1], train_indices[0], train_indices[-1]))
    elif mode == 'random':
        train_indices, dev_indices = train_test_split(indices, test_size=VALIDATION_SPLIT)
        print('mode = random sampling')
    else:
        raise AttributeError(f'Encountered unexpected mode: {mode}')
    print(f'Splitting into {len(train_indices)} trianing and {len(dev_indices)} dev samples')
    j = 0
    for i in tqdm(train_indices):
        shutil.copyfile(p.join(input_dir, f'input/{i}.npy'), p.join(output_dir, f'train/input/{j}.npy'))
        shutil.copyfile(p.join(input_dir, f'label/{i}.npy'), p.join(output_dir, f'train/label/{j}.npy'))
        j += 1
    j = 0
    for i in tqdm(dev_indices):
        shutil.copyfile(p.join(input_dir, f'input/{i}.npy'), p.join(output_dir, f'dev/input/{j}.npy'))
        shutil.copyfile(p.join(input_dir, f'label/{i}.npy'), p.join(output_dir, f'dev/label/{j}.npy'))
        j += 1

    return


def run_sampler():
    # output_dir = "/mnt/data1/yl241/datasets/ee_buffer/houston_cloud"
    # train_dev_dir = "/mnt/data1/yl241/datasets/ee_buffer/houston_train_dev_random_cloud"

    start_date = '20180101'
    num_cycles = 50
    # map_ = geemap.Map()
    # init_dir(output_dir, overwrite=False)
    # sample_region_series(start_date=start_date, num_cycles=num_cycles, input_meta=LANDSAT8_META,
    #                      label_meta=NLCD_2019_META, boundary=HOUSTON_BOUNDING_BOX, map_=map_,
    #                      output_dir=output_dir)

    # split_train_dev_local(input_dir=output_dir, output_dir=train_dev_dir, validation_split=VALIDATION_SPLIT,
    #                       mode='random')

    # print(find_next_valid_date('20211112', False))
    # print(generate_cycles(start_date='20210807', num_cycles=20))


########################################################################################

@retry(tries=10, delay=1, backoff=2)
def export_nlcd(output_dir, export_boundary, reference_landsat_img, date_):
    dataset = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD')
    nlcd2019 = dataset.filter(ee.Filter.eq('system:index', '2019')).first()
    landcover_2019 = nlcd2019.select('landcover')
    filename = os.path.join(output_dir, f'nlcd_{date_}.tif')
    try:
        projection = reference_landsat_img.projection().getInfo()
        geemap.ee_export_image(landcover_2019, filename=filename, scale=30, region=export_boundary,
                               crs=projection['crs'],
                               file_per_band=False)
    except ee.EEException as e:
        print('ERROR: ', e)
        return 1
    return 0


def build_nlcd_lux():
    decode_hex = lambda h: tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    rgb_lut = np.zeros((256, 3), dtype=np.uint8)
    for key, value in NLCD_2019_META['lut'].items():
        rgb_lut[int(key), :] = np.array(decode_hex(value))
    return rgb_lut


def color_map_nlcd(source, dest):
    nlcd = cv2.imread(source, -1)
    lut = build_nlcd_lux()
    nlcd_rgb = np.zeros_like(nlcd)
    nlcd_rgb = np.dstack((nlcd_rgb, nlcd_rgb, nlcd_rgb))
    for i in range(len(nlcd)):
        for j in range(len(nlcd[0])):
            nlcd_rgb[i, j, :] = lut[nlcd[i, j], :]

    nlcd_rgb = cv2.cvtColor(nlcd_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(dest, nlcd_rgb)
    print('NLCD RGB image written to ', dest)
    return


@retry(tries=10, delay=1, backoff=2)
def export_landsat_band(satellite, band_name, output_dir, scene_id, date_, export_boundary, affix=None,
                        scale_factor=None, offset=None):
    if affix is None:
        affix = band_name
    tier = 'TOA' if band_name[0] == 'B' else 'L2'
    img = ee.Image(f"LANDSAT/{satellite}/C02/T1_{tier}/{satellite}_{scene_id}_{date_}").select(band_name)
    filename = os.path.join(output_dir, f'{satellite}_{affix}_{date_}.tif')
    try:
        projection = img.projection().getInfo()
        if scale_factor is not None or offset is not None:
            geemap.ee_export_image(img.multiply(scale_factor).add(offset), filename=filename, scale=30, region=export_boundary,
                                   crs=projection['crs'], file_per_band=False)
        else:
            geemap.ee_export_image(img, filename=filename, scale=30,
                                   region=export_boundary,
                                   crs=projection['crs'], file_per_band=False)
    except ee.EEException as e:
        print('ERROR: ', e)
        return 1
    return 0


def export_landsat_series(output_dir, satellite, band, scene_id, export_boundary, start_date, num_cycles, affix=None,
                          getNLCD=False, scale_factor=None, offset=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print('Created directory ', output_dir)
    error_counter = 0
    acquiredNLCD = False if getNLCD else True
    cycles = generate_cycles(start_date=start_date, num_cycles=num_cycles)
    # export nlcd map using reference to the first cycle
    for date_ in tqdm(cycles):
        status = export_landsat_band(satellite=satellite, band_name=band, output_dir=output_dir, scene_id=scene_id,
                                     date_=date_, export_boundary=export_boundary, affix=affix,
                                     scale_factor=scale_factor, offset=offset)
        if status:
            error_counter += 1
        if not status and not acquiredNLCD:
            ref = ee.Image(f"LANDSAT/{satellite}/C02/T1_L2/{satellite}_{scene_id}_{date_}").select(band)
            nlcd_status = export_nlcd(output_dir, export_boundary, reference_landsat_img=ref, date_=date_)
            acquiredNLCD = 1 if nlcd_status == 0 else 0
    print(f'Missing {error_counter}/{cycles} data entries')


def resaves_bt_png(source, dest):
    """
    Saves a scaled visualization of bt and outputs16-bit PNG files.
    :param source:
    :param dest:
    :return:
    """
    low, high = 290, 320  # arbitrarily define low and high in kelvin
    assert os.path.exists(source), "ERROR: source directory does not exist"
    if not os.path.exists(dest):
        os.mkdir(dest)
    files = os.listdir(source)
    for f in tqdm(files):
        if f[-3:] != 'tif' and f[:4] != 'nlcd':
            continue
        img = cv2.imread(os.path.join(source, f), -1).astype('float32')
        img[img < 0] = 999
        img[img == 999] = img.min()
        print(f'{img.min()} | {img.max()}')
        # img = img - low
        # img /= high - low
        img -= img.min()
        img /= img.max()
        img[img >= 1] = 1
        img[img <= 0] = 0
        img *= 2 ** 16
        cv2.imwrite(os.path.join(dest, f[:-3] + 'png'), img.astype('uint16'))
    # print(f'{len([f in files and ])}')
    return


def resave_emis(source, dest):
    assert os.path.exists(source), "ERROR: source directory does not exist"
    if not os.path.exists(dest):
        os.mkdir(dest)
    files = os.listdir(source)
    for f in tqdm(files):
        if f[-3:] != 'tif' and f[:4] != 'nlcd':
            continue
        img = cv2.imread(os.path.join(source, f), -1).astype('float32')
        img *= 0.0001
        img[img > 0.9999] = 0.9999
        img[img < 0] = 0.001
        # img = img * 255
        # cv2.imwrite(os.path.join(dest, f[:-3] + 'png'), img.astype('uint8'))
        plt.imshow(img, cmap='magma', vmax=1.0, vmin=0.9)
        plt.colorbar(label='Emissivity (unitless)')
        plt.title(f'Emissivity of Houston on {f[13:21]}')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(dest, f[:-3] + 'png'))
        plt.close()
    # print(f'{len([f in files and ])}')
    return


def scale(*args):
    raise NotImplementedError


def export_rgb(output_dir, satellite, scene_id, export_boundary, start_date, num_cycles, download_monochrome=True,
               clip=None):
    """
    :param output_dir:
    :param satellite:
    :param scene_id:
    :param export_boundary:
    :param start_date:
    :param num_cycles:
    :param download_monochrome: True to download B4, B3, B2 bands
    :param clip: upper bound in pixel value for visualization
    :return:
    """
    if not p.exists(output_dir):
        os.mkdir(output_dir)

    if download_monochrome:
        export_landsat_series(pjoin(output_dir, 'B4'), satellite=satellite, band='B4', scene_id=scene_id, getNLCD=False,
                              start_date=start_date, num_cycles=num_cycles, export_boundary=export_boundary, )
        export_landsat_series(pjoin(output_dir, 'B3'), satellite=satellite, band='B3', scene_id=scene_id, getNLCD=False,
                              start_date=start_date, num_cycles=num_cycles, export_boundary=export_boundary)
        export_landsat_series(pjoin(output_dir, 'B2'), satellite=satellite, band='B2', scene_id=scene_id, getNLCD=False,
                              start_date=start_date, num_cycles=num_cycles, export_boundary=export_boundary)
    # assemble
    if not p.exists(pjoin(output_dir, 'RGB')):
        os.mkdir(pjoin(output_dir, 'RGB'))
    file_count = len([f for f in os.listdir(pjoin(output_dir, 'B4')) if f[-3:] == 'tif'])
    assert len([f for f in os.listdir(pjoin(output_dir, 'B3')) if f[-3:] == 'tif']) == file_count
    assert len([f for f in os.listdir(pjoin(output_dir, 'B2')) if f[-3:] == 'tif']) == file_count
    for r_file in tqdm(os.listdir(p.join(output_dir, 'B4'))):
        if r_file[-3:] != 'tif':
            continue
        g_file = r_file.replace('B4', 'B3')
        b_file = r_file.replace('B4', 'B2')
        r = cv2.imread(pjoin(output_dir, 'B4', r_file), -1)
        g = cv2.imread(pjoin(output_dir, 'B3', g_file), -1)
        b = cv2.imread(pjoin(output_dir, 'B2', b_file), -1)
        output = np.dstack((b, g, r))
        if clip is None:
            output_fname = r_file.replace('B4', 'RGB')
            output_fname = output_fname.replace('tif', 'png')
            output = output / output.max()
            output = output * 2 ** 16
            output = output.astype(np.uint16)
            cv2.imwrite(pjoin(output_dir, 'RGB', output_fname), output)
        else:
            print(f"clipping to range ", clip)
            output_fname = r_file.replace('B4', 'RGB')
            output_fname = output_fname.replace('tif', 'png')
            output[output > clip] = clip
            output = output / output.max() * 255
            cv2.imwrite(pjoin(output_dir, 'RGB', output_fname), output)


def parse_qa_multi(source, dest, affix, bits, threshold=None):
    """
    https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
    :param source:
    :param dest:
    :param affix:
    :param bits:
    :param threshold:
    :return:
    """
    check = lambda x, n: x & (1 << n)  # check if the nth bit in a binary seq is set
    source_filename = source[source.rindex('/') + 1:]
    assert len(bits) == 2 and bits[1] == bits[0] + 1  # can only check 2 consecutive bits

    if not os.path.exists(source):
        raise FileNotFoundError()
    if os.path.isfile(source) and source_filename[-3:] == 'tif':
        # qa_img = cv2.imread(source, -1)
        # output = check(qa_img, bits)
        # cv2.imwrite(os.path.join(dest, source_filename.replace('QA_PIXEL', affix)), output.astype('uint16'))
        raise NotImplementedError
    elif os.path.isdir(source):
        if not os.path.exists(dest):
            os.mkdir(dest)
        files = os.listdir(source)
        for f in tqdm(files):
            if f[-3:] != 'tif' or f[:4] == 'nlcd':
                continue
            qa_img = cv2.imread(os.path.join(source, f), -1)
            bit0 = check(qa_img, bits[0]) / 2 ** bits[0]
            bit1 = check(qa_img, bits[1]) / 2 ** bits[1]
            output = bit0 + 2 * bit1
            if threshold is not None:
                output[output >= threshold] = 255
            cv2.imwrite(os.path.join(dest, f.replace('QA_PIXEL', affix)), output.astype('uint8'))
    else:
        raise FileExistsError()  # source path is neither a file nor a directory
    return


def parse_qa_single(source, dest, affix, bit):
    """
    parse a single bit
    https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
    :param source:
    :param dest:
    :param affix:
    :param bit:
    :return:
    """
    check = lambda x, n: x & (1 << n)  # check if the nth bit in a binary seq is set
    source_filename = source[source.rindex('/') + 1:]
    if not os.path.exists(source):
        raise FileNotFoundError()
    if os.path.isfile(source) and source_filename[-3:] == 'tif':
        qa_img = cv2.imread(source, -1)
        output = check(qa_img, bit)
        cv2.imwrite(os.path.join(dest, source_filename.replace('QA_PIXEL', affix)), output.astype('uint16'))
    elif os.path.isdir(source):
        if not os.path.exists(dest):
            os.mkdir(dest)
        files = os.listdir(source)
        for f in tqdm(files):
            if f[-3:] != 'tif' or f[:4] == 'nlcd':
                continue
            qa_img = cv2.imread(os.path.join(source, f), -1)
            output = check(qa_img, bit)
            output[output != 0] = 255
            cv2.imwrite(os.path.join(dest, f.replace('QA_PIXEL', affix)), output.astype('uint8'))
    else:
        raise FileExistsError()  # source path is neither a file nor a directory
    return


def plot_cloud_series(root_path, city_name, scene_id, start_date, num_cycles):
    # need upgrade: currently only support LANDSAT 8
    global GLOBAL_REFERENCE_DATE
    assert p.exists(root_path)
    assert GLOBAL_REFERENCE_DATE is not None
    dates = []
    cloud_coverages = []
    cycles = generate_cycles(start_date=start_date, num_cycles=num_cycles)
    for date_ in tqdm(cycles, desc='Parsing images for cloud coverage series'):
        img_id = f"LANDSAT/LC08/C02/T1_TOA/LC08_{scene_id}_{date_}"
        try:
            img = ee.Image(img_id)
            img.getInfo()
        except ee.EEException:
            img = None
        if img is not None:
            dates.append(geemap.image_props(img).getInfo()['DATE_ACQUIRED'])
            cloud_coverages.append(geemap.image_props(img).getInfo()['CLOUD_COVER'])
        else:

            date_obj = datetime.strptime(date_, '%Y%m%d')
            date_str = date_obj.strftime('%Y-%m-%d')
            dates.append(date_str)
            cloud_coverages.append(np.nan)
    cloud_avg = np.nanmean(cloud_coverages)
    date_objects = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in dates]
    sns.set_style("darkgrid")
    sns.lineplot(x=date_objects, y=cloud_coverages)
    plt.title(f"{city_name} cloud coverages over time, avg = {cloud_avg:.2f}")
    plt.xlabel("Data Acquired")
    plt.ylabel("Cloud Cover (%)")
    plt.xticks(rotation=45)
    plt.savefig(p.join(root_path, 'cloud_coverage.png'))
    print('Cloud coverage plot saved.')
    return


@deprecated
def run_exports():
    ee.Initialize()
    # output_dir = "../data/export/bt_series"
    # export_landsat_series(output_dir, satellite='LC08', band='B10', scene_id='025039', start_date='20180101',
    #                       num_cycles=50, export_boundary=HOUSTON_BOUNDING_BOX)

    output_dir = "data/export/qa_series"
    export_landsat_series(output_dir, satellite='LC08', band='QA_PIXEL', scene_id='025039', start_date='20180101',
                          num_cycles=50, export_boundary=HOUSTON_BOUNDING_BOX)

    # print(generate_cycles(start_date='20130401', num_cycles=20))


@deprecated
def run_exports_win():
    """
    Attempting to fix display isses for TIF images on Windows Machines
    :return:
    """
    output_dir = "./data/export2/TOA_RGB"
    export_rgb(output_dir, satellite='LC08', scene_id='025039', start_date='20180101',
               num_cycles=50, export_boundary=HOUSTON_BOUNDING_BOX, download_monochrome=False, clip=0.3)


@deprecated
def export_all():
    """
    A collection that includes
    * Top-of-atmosphere RGB (TOA_RGB)
    * Brightness Temperature (bt_series)
    * Rescaled BT for visualization (bt_series_png)
    * cloud bitmask (cloud)
    * shadow bitmask (shadow)
    * cirrus (cirrus)
    * QA assessment (qa_series)
    :return:
    """
    # root_path = '../data/Houston'
    root_path = '../data/Phoenix'
    global GLOBAL_REFERENCE_DATE
    start_date = '20180101'
    cycles = 50
    # scene_id = '025039'
    # scene_id = '041036'
    scene_id = '037037'
    GLOBAL_REFERENCE_DATE = acquire_reference_date(start_date, scene_id)
    # bounding_box = HOUSTON_BOUNDING_BOX
    # LA
    # bounding_box = [[[-118.41654, 33.723626], [-118.41654, 34.333656], [-117.603448, 34.333656], [-117.603448, 33.723626], [-118.41654, 33.723626]]]
    # Phoenix
    bounding_box = [
        [[-112.39009, 33.171612], [-112.39009, 33.833492], [-111.549529, 33.833492], [-111.549529, 33.171612],
         [-112.39009, 33.171612]]]
    if not p.exists(root_path):
        os.mkdir(root_path)

    ref_img = ee.Image(f'LANDSAT/LC08/C02/T1_TOA/LC08_{scene_id}_{GLOBAL_REFERENCE_DATE}').select('B1')
    export_nlcd(root_path, bounding_box, reference_landsat_img=ref_img, date_=GLOBAL_REFERENCE_DATE)
    color_map_nlcd(source=pjoin(root_path, f'nlcd_{GLOBAL_REFERENCE_DATE}.tif'),
                   dest=pjoin(root_path, f'nlcd_{GLOBAL_REFERENCE_DATE}_color.tif'))
    export_rgb(pjoin(root_path, 'TOA_RGB'), satellite='LC08', scene_id=scene_id, start_date=start_date,
               num_cycles=cycles, export_boundary=bounding_box, download_monochrome=True, clip=0.3)
    export_landsat_series(pjoin(root_path, 'bt_series'), satellite='LC08', band='B10', scene_id=scene_id,
                          start_date=start_date, num_cycles=cycles, export_boundary=bounding_box)
    export_landsat_series(pjoin(root_path, 'qa_series'), satellite='LC08', band='QA_PIXEL', scene_id=scene_id,
                          start_date=start_date, num_cycles=cycles, export_boundary=bounding_box)
    resaves_bt_png(source=pjoin(root_path, 'bt_series'), dest=pjoin(root_path, 'bt_series_png'))
    parse_qa_single(source=pjoin(root_path, 'qa_series'), dest=pjoin(root_path, 'cirrus'), affix='cirrus', bit=2)
    parse_qa_single(source=pjoin(root_path, 'qa_series'), dest=pjoin(root_path, 'cloud'), affix='cloud', bit=3)
    parse_qa_single(source=pjoin(root_path, 'qa_series'), dest=pjoin(root_path, 'shadow'), affix='shadow', bit=4)
    return


def save_log(root_path, city_name, scene_id, ):
    raise NotImplementedError()


@time_func
def export_city(root_path, city_name, scene_id, bounding_box, high_volume_api):
    global GLOBAL_REFERENCE_DATE
    ee_init(high_volume=high_volume_api)
    start_date = '20170101'
    cycles = 125
    num_procs = 10  # number of CPU cores to be allocated, for high-volume API only
    GLOBAL_REFERENCE_DATE = acquire_reference_date(start_date, scene_id)

    # for future speed up, use a pool of threads for high-volume API
    plot_cloud_series(root_path, city_name, scene_id, start_date, cycles)
    ref_img = ee.Image(f'LANDSAT/LC08/C02/T1_TOA/LC08_{scene_id}_{GLOBAL_REFERENCE_DATE}').select('B1')
    export_nlcd(root_path, bounding_box, reference_landsat_img=ref_img, date_=GLOBAL_REFERENCE_DATE)
    color_map_nlcd(source=pjoin(root_path, f'nlcd_{GLOBAL_REFERENCE_DATE}.tif'),
                   dest=pjoin(root_path, f'nlcd_{GLOBAL_REFERENCE_DATE}_color.tif'))
    export_rgb(pjoin(root_path, 'TOA_RGB'), satellite='LC08', scene_id=scene_id, start_date=start_date,
               num_cycles=cycles, export_boundary=bounding_box, download_monochrome=True, clip=0.3)
    export_landsat_series(pjoin(root_path, 'bt_series'), satellite='LC08', band='B10', scene_id=scene_id,
                          start_date=start_date, num_cycles=cycles, export_boundary=bounding_box)
    export_landsat_series(pjoin(root_path, 'qa_series'), satellite='LC08', band='QA_PIXEL', scene_id=scene_id,
                          start_date=start_date, num_cycles=cycles, export_boundary=bounding_box)
    export_landsat_series(pjoin(root_path, 'emis'), satellite='LC08', band='ST_EMIS', scene_id=scene_id,
                          start_date=start_date, num_cycles=cycles, export_boundary=bounding_box)
    resave_emis(source=pjoin(root_path, 'emis'), dest=pjoin(root_path, 'emis_png'))
    resaves_bt_png(source=pjoin(root_path, 'bt_series'), dest=pjoin(root_path, 'bt_series_png'))
    parse_qa_single(source=pjoin(root_path, 'qa_series'), dest=pjoin(root_path, 'cirrus'), affix='cirrus', bit=2)
    parse_qa_single(source=pjoin(root_path, 'qa_series'), dest=pjoin(root_path, 'cloud'), affix='cloud', bit=3)
    parse_qa_single(source=pjoin(root_path, 'qa_series'), dest=pjoin(root_path, 'shadow'), affix='shadow', bit=4)
    generate_log(root_path=f'./data/{city_name}')
    return


def export_wrapper(city_name, high_volume_api=False, startFromScratch=True):
    if startFromScratch is False:
        rprint('WARNING: startFromScratch is disabled. Files may be over written.')
    cities_list_path = "data/us_cities.csv"
    print(f'Parsing metadata from {cities_list_path}')
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
    yprint(f'city = {city_name}, scene_id = {scene_id}, bounding_box = {bounding_box}')
    root_path = pjoin('./data', city_name)
    if p.exists(root_path):
        if startFromScratch:
            raise FileExistsError(f'Directory {root_path} already exists.')
        else:
            rprint(f'WARNING: Directory {root_path} already exists. Overriding existing files')
    else:
        os.mkdir(root_path)

    export_city(root_path, city_name, scene_id, bounding_box, high_volume_api)


def generate_log(root_path):
    """
    generates a cvs file with each row containing
    * date: date at which the satellite image is taken
    * cloud_percentage: percentage of pixels labeled as either cloud, cloud shadow
    :param root_path:
    :return:
    """
    print('Generating metadata')
    flist = os.listdir(p.join(root_path, 'cloud'))
    flist = [f for f in flist if 'tif' in f]
    dates = [f[11:-4] for f in flist]  # a list of all reference frame dates
    cloud_percentages = []
    dummy_interp = Interpolator(root=root_path, target_date=dates[0], no_log=True)
    for d in tqdm(dates, desc='scanning frames'):
        frame_valid_mask = dummy_interp.build_valid_mask(alt_date=d)
        px_count = frame_valid_mask.shape[0] * frame_valid_mask.shape[1]
        ref_percentage = np.count_nonzero(~frame_valid_mask) / px_count
        cloud_percentages.append(ref_percentage)

    df = pd.DataFrame(dates, columns=['date'])  # a list of all candidates for reference frames
    df['cloud_percentage'] = cloud_percentages
    df.to_csv(p.join(root_path, 'metadata.csv'), index=False)
    print('Meta info saved to ', p.join(root_path, 'meta_data.csv'))
    return


def add_missing_image():
    """
    After downloading data for a city via export_wrapper(), run this function should any image be
    missing. Need to manually modify variables below
    :return:
    """
    print('adding missing image...')
    ee.Initialize()
    city_name = 'Los Angeles'
    date_ = '20170812'
    band = 'ST_EMIS'
    output_dir = f'./data/{city_name}/emis/'

    # scene_id = '040037'
    # export_boundary = [[[-117.400031, 32.533427], [-116.817708, 32.533427], [-116.817708, 33.301193], [-117.400031, 33.301193], [-117.400031, 32.533427]]]
    cities_list_path = "data/us_cities.csv"
    print(f'Parsing metadata from {cities_list_path}')
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

    export_landsat_band(satellite='LC08', band_name=band, output_dir=output_dir, scene_id=scene_id,
                        date_=date_, export_boundary=bounding_box)


def main():
    parser = argparse.ArgumentParser(description='Process specify city name.')
    parser.add_argument('-c', nargs='+', required=True,
                        help='Process specify city name.')
    args = parser.parse_args()
    CITY_NAME = ""
    for entry in args.c:
        CITY_NAME += entry + " "
    CITY_NAME = CITY_NAME[:-1]
    # CITY_NAME = 'San Diego'
    wandb.init()
    export_wrapper(city_name=CITY_NAME, high_volume_api=True, startFromScratch=False)
    # generate_log(root_path=f'../data/{CITY_NAME}')
    wandb.alert(
        title='Download finished',
        text=f'Data for region {CITY_NAME} finished downloading.'
    )


if __name__ == '__main__':
    # main()
    add_missing_image()

# single-program: Processing time = 0:20:36.844000
# Phoenix, standard API, 23 min
# Phoenix, high-volume API, 19 min
