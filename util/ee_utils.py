__author__ = 'yuhao liu'
"""
Earth Engine related utilities
"""
import os
import ee
from datetime import datetime


def get_landsat_lst(lon, lat, image=None, url=None):
    """
    Get the land surface temperature (LST) from a Landsat image at a specific location.
    There is no consideration on cloud occlusion.
    :param lon:
    :param lat:
    :param image: unscaled Landsat image
    :param url: url of Landsat image
    :return:
    """
    if ee.Image is not None and url is None:
        assert isinstance(image, ee.Image), "image must be an ee.Image object"
        pass
    elif url is not None and ee.Image is None:
        assert isinstance(url, str), "url must be a string"
        image = ee.Image(url)
    else:
        raise ValueError("Either image or url must be provided")
    point = ee.Geometry.Point(lon, lat)
    image = image.multiply(0.00341802).add(149)
    info = image.reduceRegion(ee.Reducer.first(), point, scale=30).getInfo()
    lst = info['ST_B10']
    return lst


def get_landsat_capture_time(image=None, url=None):
    if ee.Image is not None and url is None:
        assert isinstance(image, ee.Image), "image must be an ee.Image object"
        pass
    elif url is not None and ee.Image is None:
        assert isinstance(url, str), "url must be a string"
        image = ee.Image(url)
    else:
        raise ValueError("Either image or url must be provided")
    # Get the capture time
    capture_time = image.get('system:time_start')
    # Retrieve the value from the server
    capture_time_info = capture_time.getInfo()
    # Convert the timestamp to minutes, round it, and then convert back to milliseconds
    rounded_time = ee.Number(capture_time_info).divide(1000 * 60).round().multiply(1000 * 60)
    # Convert the rounded timestamp to a readable date
    rounded_date = ee.Date(rounded_time).format('YYYY-MM-dd HH:mm').getInfo()
    dt = datetime.strptime(rounded_date, '%Y-%m-%d %H:%M')
    return dt
