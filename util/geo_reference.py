__author__ = 'Pranavesh Panakkal'
# Title     : Geo-reference a jpg file give lat long at corners
# Objective :
# Created by:
# Created on: 1/24/23
# Version   : 0.1

import numpy as np
import rasterio
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import cv2


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


if __name__ == "__main__":
    # ordered_coors = [[-95.690165, 29.5937], [-95.690165, 30.266005], [-94.900379, 30.266005], [-94.900379, 29.5937]]
    # ordered_coors = [[[-95.690165, 29.5937], [-95.690165, 30.266005], [-94.900379, 30.266005],
    #                   [-94.900379, 29.5937], [-95.690165, 29.5937]]]
    phoenix = [[[-112.39009, 33.171612], [-112.39009, 33.833492], [-111.549529, 33.833492], [-111.549529, 33.171612], [-112.39009, 33.171612]]]
    # path_image_data = '../data/Houston/bt_series_png/LC08_B10_20180103.png'
    phoenix_nlcd = '../data/Phoenix/nlcd_20170104.tif'
    results_tiff_file_path = '../tmp/phoenix_nlcd.tif'
    geo_ref(phoenix, phoenix_nlcd, results_tiff_file_path)
    print('file saved.')
