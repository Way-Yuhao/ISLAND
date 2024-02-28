__author__ = 'yuhao liu'
"""
Earth Engine related utilities
"""
import os
import ee
from datetime import datetime, date, timedelta


def load_ee_image(url):
    """
    Safely loads an image from a URL and immediate checks if image exists. Throws ee.EEException if image does not exist.
    :param url:
    :return:
    """
    image = ee.Image(url)
    image.getInfo()
    return image


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


def is_landsat_pixel_clear(lon, lat, image=None, url=None):
    """
    Check if a pixel in a Landsat image is clear of clouds using CFMask.
    :param lon:
    :param lat:
    :param image: unscaled Landsat image
    :param url: url of Landsat image
    :return: True if the pixel is clear of clouds and dilated clouds, False otherwise
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
    cloud_mask = image.select('QA_PIXEL')
    pixel_value = cloud_mask.reduceRegion(ee.Reducer.first(), point, scale=30).getInfo()['QA_PIXEL']
    clear_mask = 1 << 6
    return (pixel_value & clear_mask) != 0


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


def acquire_reference_date(start_date, scene_id):
    """
    Find the first date after which a LANDSAT 8 image is valid for a given scene id
    :param start_date:
    :param scene_id:
    :return:
    """
    ee.Initialize()
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


def generate_cycles(start_date, end_date):
    """
    Generate the cycles of 8-day intervals for the given start and end date
    :param start_date: YYYYMMDD, assumed to be the first day of a cycle
    :param end_date:
    :return:
    """
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    cur = start
    cycles = []
    while cur < end:
        # append the current cycle in YYYYMMDD format
        cycles.append(cur.strftime('%Y%m%d'))
        cur = cur + timedelta(days=16)
    return cycles
