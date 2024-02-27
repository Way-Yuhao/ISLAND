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
from io import StringIO


def read_surfrad_file_from_url(config, url):
    """
    Read a SURFRAD data file from a URL and return a DataFrame.
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
        # Add station metadata as new columns to the DataFrame
        df['station_name'] = station_name
        df['latitude'] = latitude
        df['longitude'] = longitude
        df['elevation'] = elevation
        return df
    else:
        print(f"Failed to download the file: HTTP {response.status_code}")
        return None


def get_emis_at(lon, lat):
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


def calc_broadband_emis(emis_10, emis_11, emis_12, emis_13, emis_14):
    """
    Calculate the broadband emissivity from the ASTER emissivity bands.
    Following K. Ogawa, T. Schmugge, and S. Rokugawa, “Estimating broadband emissivity of arid regions and its
    seasonal variations using thermal infrared remote sensing,” (in English),
    IEEE Trans. Geosci. Remote Sens., vol. 46, no. 2, pp. 334–343, Feb. 2008.
    :param emis_10:
    :param emis_11:
    :param emis_12:
    :param emis_13:
    :param emis_14:
    :return:
    """
    # Calculate the broadband emissivity
    broadband_emis = 0.128 + 0.014 * emis_10 + 0.145 * emis_11 + 0.241 * emis_12 + 0.467 * emis_13 + 0.004 * emis_14
    return broadband_emis

@hydra.main(version_base=None, config_path='../config', config_name='surfrad.yaml')
def main(surfrad_config: DictConfig):
    # Example usage
    url = 'https://gml.noaa.gov/aftp/data/radiation/surfrad/psu/2020/psu20184.dat'
    df = read_surfrad_file_from_url(surfrad_config, url)
    if df is not None:
        print(df.head())


if __name__ == '__main__':
    main()