
"""
This module is used to download and process the data from SURFRAD website.
"""
__author__ = 'yuhao liu'

import os
import os.path as p
import sys
import ee
from omegaconf import DictConfig, OmegaConf
import hydra
import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from io import StringIO
import geemap
from rich.progress import Progress
from util.equations import calc_lst, cvt_celsius_to_kelvin, calc_broadband_emis_ogawa, calc__broadband_emis_cheng
from util.ee_utils import (acquire_reference_date, generate_cycles,
                           get_landsat_lst, get_landsat_capture_time, load_ee_image,
                           is_landsat_pixel_clear, query_geotiff, get_landsat_cloud_percentage)
from util.helper import rprint, yprint


def correct_station_id(station_id):
    """
    Corrects a naming incosistency in the SURFRAD station IDs.
    :param station_id:
    :return:
    """
    if station_id == 'BND':
        station_id = 'BON'
    return station_id


def read_surfrad_file_from_url(config, url):
    """
    Read a SURFRAD data file from a URL and return a DataFrame.
    * Do NOT use lon, lat and elevation from the file, as they are not accurate.
    Example url: 'https://gml.noaa.gov/aftp/data/radiation/surfrad/psu/2020/psu20184.dat'
    :param url:
    :return:
    """
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Use StringIO to convert the text to a file-like object
        file = StringIO(response.text)
        # Read the station name (first line)
        station_name = file.readline().strip()
        # Attempt to read the latitude, longitude, and elevation (second line)
        # Skipping non-numeric parts (like 'm' for meters) by filtering the split parts
        lat_lon_ele_line = file.readline().split()
        # Assuming latitude, longitude, and elevation are always the first three floats in the line
        latitude, longitude, elevation = map(float,
                                             [i for i in lat_lon_ele_line if i.replace('.', '', 1).isdigit()][:3])
        # Initialize a list to store the data rows
        data_rows = []
        # Read each line of data
        for line in file:
            values = line.split()
            if len(values) < 48:  # or another appropriate check for line completeness
                continue
            try:
                # Assuming the first few fields are integers (year, jday, month, day, hour, min)
                year, jday, month, day, hour, min = map(int, values[:6])
                # The remaining values will be processed as floats including QC values for simplicity.
                # Adjust this if you have a clear distinction between float and int fields.
                rest_values = list(map(float, values[6:]))
                row = [year, jday, month, day, hour, min] + rest_values
                data_rows.append(row)
            except ValueError as e:
                print(f"Error processing line: {e}")
                continue
        df = pd.DataFrame(data_rows, columns=config.column_names)
        # Convert each identified QC column to integer
        qc_columns = [col for col in df.columns if col.startswith('qc_')]
        for col in qc_columns:
            df[col] = df[col].astype(int)
        return df
    else:
        print(f"Failed to download the file: HTTP {response.status_code}")
        return None


def get_emis_at(lon, lat, use_equation='ogawa'):
    """
    Get the broadband emissivity at a given location using the ASTER GEDv3 dataset.
    :param lon:
    :param lat:
    :return:
    """
    point = ee.Geometry.Point(lon, lat)
    image = ee.Image('NASA/ASTER_GED/AG100_003').multiply(0.001)
    info = image.reduceRegion(ee.Reducer.first(), point, scale=30).getInfo()
    emis_10 = info['emissivity_band10']
    emis_11 = info['emissivity_band11']
    emis_12 = info['emissivity_band12']
    emis_13 = info['emissivity_band13']
    emis_14 = info['emissivity_band14']
    if use_equation == 'cheng':
        broadband_emis = calc__broadband_emis_cheng(emis_10, emis_11, emis_12, emis_13, emis_14)
    elif use_equation == 'ogawa':
        broadband_emis = calc_broadband_emis_ogawa(emis_10, emis_11, emis_12, emis_13, emis_14)
    else:
        raise ValueError(f"Invalid use_equation: {use_equation}")
    return broadband_emis


def run_qc_check(row, *args):
    for arg in args:
        if row[arg].iloc[0] != 0:
            return False
    return True


def get_surfrad_surf_temp_at(station_id: str, time: datetime, qc_check: bool = True, use_emis_from: str = 'ogawa'):
    """
    Get the surface temperature at a given SURFRAD station and time.
    May throw a ValueError if no data is available for the given time.
    :param station_id:
    :param time:
    :return:
    """
    assert isinstance(time, datetime), "time must be a datetime object"
    # correct_station_id(station_id)
    year_ = time.year
    year_2 = str(time.year)[-2:]
    jday = str(time.timetuple().tm_yday).zfill(3)
    url = f'https://gml.noaa.gov/aftp/data/radiation/surfrad/{correct_station_id(station_id).lower()}/{year_}/{correct_station_id(station_id).lower()}{year_2}{jday}.dat'
    config = OmegaConf.load('../config/surfrad.yaml')
    if use_emis_from == 'ogawa':
        emis = config['stations'][station_id]['emis']
    elif use_emis_from == 'cheng':
        emis = config['boradband_emis_cheng'][station_id]
    # find df for the corresponding day
    df = read_surfrad_file_from_url(config, url)
    assert df is not None, f'Failed to read the SURFRAD data from {url}'
    # find data for the corresponding time during the day
    row = df[df['hour'] == time.hour]
    row = row[row['min'] == time.minute]
    if len(row) == 0:
        raise ValueError(f"No data for the given time {time} at station {station_id}")
    if qc_check:
        qc = run_qc_check(row, 'qc_uwir', 'qc_dwir') # not doing qc check for air temperature
        if qc is False:
            raise ValueError(f"QC check failed for the given time {time} at station {station_id}")
    uw_ir = row['uw_ir'].iloc[0]
    dw_ir = row['dw_ir'].iloc[0]
    air_temp = row['temp'].iloc[0]  # in Celcius
    air_temp = cvt_celsius_to_kelvin(air_temp)  # in Kelvin
    try:
        surf_temp = calc_lst(emis, uw_ir, dw_ir)
    except RuntimeWarning as e:  # unable to catch RuntimeWarning ....
        print(e)
        print(f"Failed to calculate the surface temperature at station {station_id} and time {time}")
        print(f'uw_ir: {uw_ir}, dw_ir: {dw_ir}, air_temp: {air_temp}')
        surf_temp = np.nan
    return {'surf_temp': surf_temp, 'air_temp': air_temp}



######################## time series plot ########################


def load_datapoints(station_id: str, start_date: str, end_date: str, region_dir,
                    use_emis_from: str = 'ogawa', load_from_cache: bool = False,
                    mode: str = 'full') -> pd.DataFrame:
    """
    Load the data points for a given SURFRAD station and time range.
    :param station_id:
    :param start_date:
    :param end_date:
    :param region_dir:
    :param use_emis_from: 'ogawa' (consistent with landsat 8 manual) or 'cheng'
    :return:
    """
    config = OmegaConf.load('../config/surfrad.yaml')
    lon = config['stations'][station_id]['Longitude']
    lat = config['stations'][station_id]['Latitude']
    scene_id = config['stations'][station_id]['scene_id']

    if region_dir is None:
        read_local_files = False
        print('No region_dir is specified. Only reading data from Google Earth Engine.')
    else:
        read_local_files = True
        assert p.exists(region_dir), f"region_dir {region_dir} does not exist"
        print(f"Reading local data from {region_dir}")

    if mode == 'full':
        df_path = p.join(region_dir, f'surfrad_datapoints.csv')
        rprint('Using full mode')
    elif mode == 'temporal':
        df_path = p.join(region_dir, f'surfrad_datapoints_temporal.csv')
        rprint('Using temporal mode')
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if load_from_cache and p.exists(df_path):  # load from cache
        df = pd.read_csv(df_path)
        assert df is not None, f"Failed to load the DataFrame from {df_path}"
        print(f"Loaded the DataFrame from {df_path}")
    else:  # fetch from server
        ref_date = acquire_reference_date(start_date, scene_id)
        cycles = generate_cycles(ref_date, end_date)
        data = []
        with Progress() as progress:
            task_id = progress.add_task("[cyan]Processing...", total=len(cycles))
            for date_ in cycles:
                try:
                    image = load_ee_image(f'LANDSAT/LC08/C02/T1_L2/LC08_{scene_id}_{date_}')
                    landsat_lst = get_landsat_lst(lon, lat, image=image)
                    capture_time = get_landsat_capture_time(image=image)
                    cloud_perc = get_landsat_cloud_percentage(capture_time=capture_time, region_dir=region_dir)
                    surfrad_data = get_surfrad_surf_temp_at(station_id, capture_time,
                                                            qc_check=True, use_emis_from=use_emis_from)
                    surfrad_lst, surfrad_air_temp = surfrad_data['surf_temp'], surfrad_data['air_temp']
                    if read_local_files:
                        img_path = p.join(region_dir, f'lst/LC08_ST_B10_{date_}.tif')
                        download_lst = query_geotiff(lon, lat, img_path)
                        if mode == 'full':
                            island_path = p.join(region_dir, f'output_referenced/lst/lst_{date_}.tif')
                        elif mode == 'temporal':
                            island_path = p.join(region_dir, f'output_referenced_temporal/lst/lst_{date_}.tif')
                        island_lst = query_geotiff(lon, lat, island_path)
                    else:
                        download_lst = None
                        island_lst = None
                    condition_clear = is_landsat_pixel_clear(lon, lat, image=image)
                    data.append({
                        'date': date_,
                        'landsat_lst': landsat_lst,
                        'surfrad_lst': surfrad_lst,
                        'download_lst': download_lst,
                        'island_lst': island_lst,
                        'surfrad_air_temp': surfrad_air_temp,
                        'condition_clear': condition_clear,
                        'cloud_perc': cloud_perc,
                        'error': island_lst - surfrad_lst
                    })
                except (ee.EEException, ValueError) as e:
                    continue
                progress.update(task_id, advance=1)
            progress.update(task_id, completed=True, description="Datapoints loaded.")
        df = pd.DataFrame(data)
        # save the DataFrame to a csv file
        df.to_csv(df_path, index=False)
        print(f"Saved the DataFrame to {df_path}")
    return df


def plot_regression(df: pd.DataFrame, x_label: str, y_label: str, select_condition: str, title: str,
                    fig_path=None, replace_axis=None) -> None:
    """
    Plot a regression plot for the given DataFrame.
    :param df:
    :param x_label:
    :param y_label:
    :param title:
    :param fig_path:
    :return:
    """
    # Filter the DataFrame
    if select_condition == 'clear':
        df_filtered = df[df['condition_clear']]
    elif select_condition == 'cloudy':
        df_filtered = df[df['condition_clear'] == False]
        df_filtered = df_filtered[df_filtered['island_lst'] > 0]
    elif select_condition == 'all':
        df_filtered = df
    else:
        raise ValueError(f"Invalid select_condition: {select_condition}")

    # Create a scatter plot
    sns.set_context("paper", font_scale=2.25)
    plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    # Add a dashed black line y = x
    x = range(240, 340)
    plt.plot(x, x, color='black', linestyle='--', linewidth=0.5)
    sns.scatterplot(y=y_label, x=x_label, data=df_filtered, edgecolor='none')
    # Set x and y limits
    plt.xlim(240, 340)
    plt.ylim(240, 340)
    plt.title(title)
    if replace_axis is not None:
        plt.xlabel(replace_axis['x_label'])
        plt.ylabel(replace_axis['y_label'])

    # computing errors
    actual = df_filtered[x_label]
    predicted = df_filtered[y_label]
    errors = predicted - actual
    rmse = np.sqrt((errors ** 2).mean())
    bias = errors.mean()
    mae = mean_absolute_error(actual, predicted)
    print(f"RMSE: {rmse:.2f} K, Bias: {bias:.2f} K, MAE: {mae:.2f} K")
    print('Num =', len(df_filtered))

    # Add text to the plot
    plt.text(245, 310,
             f"RMSE: {rmse:.2f} K\nBias: {bias:.2f} K\nMAE: {mae:.2f} K\nNum: {len(df_filtered)}",
             fontsize=14)

    if fig_path is not None:
        plt.tight_layout()
        plt.savefig(p.join(fig_path, f'{title}.svg'), format='svg', dpi=300)
        print(f"Saved the figure to {fig_path}")
    plt.show()
    return


def cache_all_datapoints():
    start_date = '20130411'
    end_date = '20201231'
    config = OmegaConf.load('../config/surfrad.yaml')
    for station_id in config.station_list:
        region_dir = f'/home/yuhaoliu/Data/ISLAND/surfrad_val/{station_id}'
        load_datapoints(station_id, start_date, end_date, region_dir)


if __name__ == '__main__':
    cache_all_datapoints()
    # geemap.Map()
    # station_id = 'BND'
    # scene_id = '023032'
    # date_ = '20150521'
    # image = load_ee_image(f'LANDSAT/LC08/C02/T1_L2/LC08_{scene_id}_{date_}')
    # capture_time = get_landsat_capture_time(image=image)
    # surfrad_lst = get_surfrad_surf_temp_at(station_id, capture_time, qc_check=True)
    # print(surfrad_lst)

    # station_id = 'BND'
    # region_dir = f'/home/yuhaoliu/Data/ISLAND/surfrad_val/BND'
    # start_date = '20130411'
    # end_date = '20201231'
    # load_datapoints(station_id, start_date, end_date, region_dir)
