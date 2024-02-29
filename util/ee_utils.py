__author__ = 'yuhao liu'
"""
Earth Engine related utilities
"""
import os
import ee
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points


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


def cvt_lat_lon_to_path_row(lat: float, lon: float) -> dict:
    """
    Convert latitude and longitude to Landsat WRS-2 Path and Row.
    If there are multiple matches, the patch whose center is closest to the specified point is returned.
    :param lat: latitude
    :param lon: longitude
    :return: dictionary with 'path' and 'row'
    """
    # Load WRS-2 Path/Row shapefile
    wrs_path_row_shp = '../config/WRS2/WRS2_descending.shp'
    assert os.path.exists(wrs_path_row_shp), f'File not found: {wrs_path_row_shp}'
    wrs = gpd.read_file('../config/WRS2/WRS2_descending.shp')
    point = Point(lon, lat)
    # Perform spatial query to find the Path/Row intersecting with the point
    matches = wrs[wrs.geometry.intersects(point)]
    if len(matches) == 0 or matches is None:
        print('No matching Path/Row found.')
        return None
    else:
        # Define a projected CRS that is appropriate for your area of interest
        # This is an example using EPSG:3857 which is commonly used for Web Mercator
        projected_crs = "EPSG:3857"
        # Convert the CRS of the geometries
        matches = matches.copy()
        matches.geometry = matches.geometry.to_crs(projected_crs)
        point = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(projected_crs)[0]
        # Now you can calculate the centroid and distance
        distances = matches.geometry.centroid.distance(point)
        # Find the index of the patch with the smallest distance
        closest_index = distances.idxmin()
        # Get the path and row of the closest patch
        closest_path = matches.loc[closest_index, 'PATH']
        closest_row = matches.loc[closest_index, 'ROW']
        return {'path': closest_path, 'row': closest_row}


if __name__ == '__main__':
    # Test the function cvt_lat_lon_to_path_row
    lon = -95.096
    lat = 31.230
    patch = cvt_lat_lon_to_path_row(lat, lon)
    print(patch)