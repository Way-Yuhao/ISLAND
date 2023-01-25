__author__ = 'Pranavesh Panakkal'
# Title     : Geo-reference a jpg file give lat long at corners
# Objective :
# Created by:
# Created on: 1/24/23
# Version   : 0.1

# %% Import
import numpy as np
import rasterio
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr


# %% Function
def get_georeferenced_tiff_from_npy(ordered_coors: list, path_image_data: str, results_tiff_file_path: str,
                                    crs: int = 4236) -> None:
    """
    This code will generate geo-referenced tiff file from numpy dataset
    :param ordered_coors: a list of coords arranged in the form of [bottom-left, top-left, top-right, bottom-right].
                           Eg: [[-95.690165, 29.5937], ..., ..., [-94.900379, 29.5937]]
    :param path_image_data: path to image numpy data
    :param results_tiff_file_path: file path to save the results
    :param crs: epsg code
    :return: None
    :notes: Adapted from https://gis.stackexchange.com/a/37431
    """
    # Get control points
    xmin, ymin, xmax, ymax = [ordered_coors[0][0], ordered_coors[0][1], ordered_coors[-2][0], ordered_coors[-2][1]]
    # Read numpy data
    data = np.load(path_image_data)
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


if __name__ == "__main__":
    ordered_coors = [[-95.690165, 29.5937], [-95.690165, 30.266005], [-94.900379, 30.266005], [-94.900379, 29.5937]]
    path_image_data = '/Users/pranavesh/Documents/research/13_landsat/01_Data/hotspots_aggregate.npy'
    results_tiff_file_path = 'geo_referenced_image.tiff'
    get_georeferenced_tiff_from_npy(ordered_coors, path_image_data, results_tiff_file_path)