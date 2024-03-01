import rasterio
from rasterio.transform import from_origin
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pyproj import Transformer

import numpy as np
import rasterio
from pyproj import Transformer

def query_geotiff(lon, lat, geotiff_path):
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as img:
        # Initialize transformer to convert from WGS 84 (EPSG:4326) to EPSG:32613
        transformer = Transformer.from_crs("EPSG:4326", img.crs.to_string(), always_xy=True)
        # Transform the coordinates
        x, y = transformer.transform(lon, lat)
        # Since GeoTIFF might have multiple bands, assuming you're interested in the first band
        band1 = img.read(1)
        # Get the pixel values' row and column in the dataset
        row, col = img.index(x, y)
        # Query the pixel value using row and column
        pixel_value = band1[row, col]
    return pixel_value


config = OmegaConf.load('../config/surfrad.yaml')
lon = -97.710
lat = 30.338

# Example usage
geotiff_path = '/home/yuhaoliu/Code/UrbanSurfTemp/data/Austin/lst/LC08_ST_B10_20170303.tif'
pixel_value = query_geotiff(lon, lat, geotiff_path)
print("Pixel value:", pixel_value)