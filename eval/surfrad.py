
"""
This module is used to download and process the data from SURFRAD website.
"""
__author__ = 'yuhao liu'

import os
import sys
import ee
from omegaconf import DictConfig, OmegaConf
import hydra
import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from util.equations import calc_lst, calc_broadband_emis


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
        # DO NOT USE station metadata as new columns to the DataFrame
        # df['station_name'] = station_name
        # df['latitude'] = latitude
        # df['longitude'] = longitude
        # df['elevation'] = elevation
        return df
    else:
        print(f"Failed to download the file: HTTP {response.status_code}")
        return None


def get_emis_at(lon, lat):
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
    broadband_emis = calc_broadband_emis(emis_10, emis_11, emis_12, emis_13, emis_14)
    return broadband_emis


def get_surfrad_surf_temp_at(station_id: str, time: datetime):
    assert isinstance(time, datetime), "time must be a datetime object"
    year_ = time.year
    year_2 = str(time.year)[-2:]
    jday = str(time.timetuple().tm_yday).zfill(3)
    url = f'https://gml.noaa.gov/aftp/data/radiation/surfrad/{station_id.lower()}/{year_}/{station_id.lower()}{year_2}{jday}.dat'
    config = OmegaConf.load('../config/surfrad.yaml')
    emis = config['stations'][station_id]['emis']
    # find df for the corresponding day
    df = read_surfrad_file_from_url(config, url)
    assert df is not None, f'Failed to read the SURFRAD data from {url}'
    # find data for the corresponding time during the day
    row = df[df['hour'] == time.hour]
    row = row[row['min'] == time.minute]
    uw_ir = row['uw_ir'].iloc[0]
    dw_ir = row['dw_ir'].iloc[0]
    air_temp = row['temp'].iloc[0]
    # print('upwelling thermal infrared (Watts m^-2): ', uw_ir)
    # print('downwelling thermal infrared (Watts m^-2): ', dw_ir)
    # print('10-m air temperature (Celcius): ', air_temp)
    surf_temp = calc_lst(emis, uw_ir, dw_ir)
    return surf_temp




@hydra.main(version_base=None, config_path='../config', config_name='surfrad.yaml')
def main(surfrad_config: DictConfig):
    # Example usage
    url = 'https://gml.noaa.gov/aftp/data/radiation/surfrad/psu/2020/psu20184.dat'
    df = read_surfrad_file_from_url(surfrad_config, url)
    if df is not None:
        print(df.head())


if __name__ == '__main__':
    # main()
    lst = get_surfrad_surf_temp_at('BND', datetime(2017, 2, 3, 16, 36))
    print(lst)