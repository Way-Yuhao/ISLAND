"""
This module is used to download and process the data from SURFRAD website.
"""
__author__ = 'yuhao liu'

import os
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
import requests
import pandas as pd
from io import StringIO

column_names = [
            'year', 'jday', 'month', 'day', 'hour', 'min', 'dt', 'zen',
            'dw_solar', 'uw_solar', 'direct_n', 'diffuse',
            'dw_ir', 'dw_casetemp', 'dw_dometemp',
            'uw_ir', 'uw_casetemp', 'uw_dometemp',
            'uvb', 'par', 'netsolar', 'netir', 'totalnet',
            'temp', 'rh', 'windspd', 'winddir', 'pressure'
        ]


def load_file_from_url(url):
    """Load a file from a given URL, excluding the first two lines in the DataFrame."""

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Separate the content into lines
        lines = response.text.splitlines()

        # Store the first two lines as metadata
        metadata = "\n".join(lines[:2])

        # Load the rest of the data into a DataFrame, skipping the first two lines
        from io import StringIO
        data = StringIO("\n".join(lines[2:]))
        df = pd.read_csv(data, names=column_names, header=None)

        return metadata, df
    else:
        print(f"Failed to download the file: HTTP {response.status_code}")
        return None, None


def read_surfrad_file_from_url(url):
    # Send a GET request to the URL
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

        # Define the column names based on the Fortran code variables and their quality control indicators
        column_names = [
            'year', 'jday', 'month', 'day', 'hour', 'min',
            'dt', 'zen', 'dw_solar', 'qc_dwsolar', 'uw_solar', 'qc_uwsolar', 'direct_n',
            'qc_direct_n', 'diffuse', 'qc_diffuse', 'dw_ir', 'qc_dwir', 'dw_casetemp',
            'qc_dwcasetemp', 'dw_dometemp', 'qc_dwdometemp', 'uw_ir', 'qc_uwir', 'uw_casetemp',
            'qc_uwcasetemp', 'uw_dometemp', 'qc_uwdometemp', 'uvb', 'qc_uvb', 'par', 'qc_par',
            'netsolar', 'qc_netsolar', 'netir', 'qc_netir', 'totalnet', 'qc_totalnet', 'temp',
            'qc_temp', 'rh', 'qc_rh', 'windspd', 'qc_windspd', 'winddir', 'qc_winddir',
            'pressure', 'qc_pressure'
        ]

        # Create a DataFrame from the data rows
        df = pd.DataFrame(data_rows, columns=column_names)

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


def main():
    # Example usage
    url = 'https://gml.noaa.gov/aftp/data/radiation/surfrad/psu/2020/psu20184.dat'
    df = read_surfrad_file_from_url(url)
    if df is not None:
        print(df.head())


if __name__ == '__main__':
    main()