__author__ = 'Pranavesh Panakkal'
# Title     : Geo-reference a jpg file give lat long at corners
# Objective :
# Created by:
# Created on: 1/24/23
# Version   : 0.1

import os
import os.path as p
import numpy as np
import rasterio
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import cv2
from util.helper import deprecated

@deprecated
def geo_ref(ordered_coors: list, path_image_data: str, results_tiff_file_path: str,
            crs: int = 4326) -> None:
    """
    This code will generate geo-referenced tiff file from numpy dataset
    :param ordered_coors: a list of coords arranged in the form of [bottom-left, top-left, top-right, bottom-right].
                           Eg: [[-95.690165, 29.5937], ..., ..., [-94.900379, 29.5937]]
    :param path_image_data: path to image data. Supports npy or files readable via cv2
    :param results_tiff_file_path: file path to save the results
    :param crs: epsg code
    :return: None
    :notes: Adapted from https://gis.stackexchange.com/a/37431
    """
    # Get control points
    ordered_coors = np.array(ordered_coors)
    shape_ = ordered_coors.shape
    if len(shape_) == 3 and shape_[0] == 1:
        ordered_coors = ordered_coors[0]
        shape_ = ordered_coors.shape
    if len(shape_) == 2:
        if shape_[0] == 4:
            pass
        elif shape_[0] == 5:
            ordered_coors = ordered_coors[:-1]
        else:
            raise IndexError()
    else:
        raise IndexError()
    xmin, ymin, xmax, ymax = [ordered_coors[0][0], ordered_coors[0][1], ordered_coors[-2][0], ordered_coors[-2][1]]
    # Read numpy data
    file_type = path_image_data[-4:]
    if 'npy' in file_type:
        data = np.load(path_image_data)
    elif 'tif' in file_type or 'jpg' in file_type or 'png' in file_type:
        data = cv2.imread(path_image_data, -1)
    else:
        raise FileNotFoundError()
    # Get image shape
    nrows, ncols = np.shape(data)
    # Estimate resolution
    xres = (xmax - xmin) / float(ncols)
    yres = (ymax - ymin) / float(nrows)
    # Define transform
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    # Generate output raster placeholder
    output_raster = gdal.GetDriverByName('GTiff').Create(results_tiff_file_path, ncols, nrows, 1,
                                                         gdal.GDT_Float32)
    # Specify its coordinates
    output_raster.SetGeoTransform(geotransform)
    # Establish its coordinate encoding
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs)
    # Exports the coordinate system
    output_raster.SetProjection(srs.ExportToWkt())
    # Writes the array to the raster
    output_raster.GetRasterBand(1).WriteArray(data)
    # Flush cache
    output_raster.FlushCache()
    return


def geo_ref_copy(city, npy_filename, out_path='default'):
    """
    Copies geo-reference data from corresponding GeoTIFF input files and past to output
    :param city:
    :param npy_filename: input filename. Requires file to be stored in .../{city}/output_XX_/npy/ with a format
    of XX_YYYYMMDD.npy, where XX is mode and Y M D are year, month, date, respectively.
    :param out_path:
    :return:
    """
    root_ = f'./data/{city}'
    mode = npy_filename[:2]
    date_ = npy_filename[3:11]
    npy_path = p.join(root_, f'output_{mode}/npy', npy_filename)
    assert p.exists(npy_path)
    npy_content = np.load(npy_path)
    assert npy_content is not None
    reference_geotiff_path = p.join(root_, f'bt_series/LC08_B10_{date_}.tif')
    assert p.exists(reference_geotiff_path), \
        f'Reference file {reference_geotiff_path} does not exist'
    ref_img = rasterio.open(reference_geotiff_path)
    if out_path == 'default':
        p.join(root_, f'output_referenced/{mode}/{mode}_{date_}.tif')
    out_tif = rasterio.open(out_path, 'w',
                            driver='Gtiff', height=ref_img.height, width=ref_img.width,
                            count=1, crs=ref_img.crs, transform=ref_img.transform,
                            dtype=npy_content.dtype)
    out_tif.write(npy_content, 1)
    ref_img.close()
    out_tif.close()


def save_geotiff(city, img, date_, out_path):
    """

    :param city:
    :param img: numpy array file
    :param date_:
    :param out_path:
    :return:
    """
    root_ = f'./data/{city}'
    reference_geotiff_path = p.join(root_, f'bt_series/LC08_B10_{date_}.tif')
    ref_img = rasterio.open(reference_geotiff_path)
    out_tif = rasterio.open(out_path, 'w',
                            driver='Gtiff', height=ref_img.height, width=ref_img.width,
                            count=1, crs=ref_img.crs, transform=ref_img.transform,
                            dtype=img.dtype)
    out_tif.write(img, 1)
    ref_img.close()
    out_tif.close()
    return


if __name__ == "__main__":
    # geo_ref_copy('Houston', 'bt_20170116.npy', 'default')
    os.chdir('..')
    date_ = '20170422'
    city = 'Houston'
    cloud = f'./data/{city}/cloud/LC08_cloud_{date_}.tif'
    shadow = f'./data/{city}/shadow/LC08_shadow_{date_}.tif'
    # assert p.exists(cloud)
    cloud_img = cv2.imread(cloud, -1)
    shadow_img = cv2.imread(shadow, -1)
    occlusion = cloud_img + shadow_img
    occlusion[occlusion > 255] = 255
    print(occlusion)
    save_geotiff(city, occlusion, date_, out_path=f'./data/{city}/analysis/LC08_occlusion_{date_}.tif')