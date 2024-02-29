import rasterio
from rasterio.transform import from_origin
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pyproj import Transformer

def query_geotiff(lon, lat, geotiff_path):
    """
    Query a GeoTIFF image at a specific longitude and latitude.

    Args:
        lon (float): Longitude to query.
        lat (float): Latitude to query.
        geotiff_path (str): Path to the GeoTIFF file.

    Returns:
        The pixel value at the specified longitude and latitude, or None if
        the coordinates are outside the image bounds.
    """
    with rasterio.open(geotiff_path) as src:
        print(src.bounds)
        # Convert the longitude and latitude to pixel coordinates
        py, px = src.index(lon, lat)
        print(py, px)

        # Read the pixel value at the calculated position
        try:
            # Window of 1x1 pixel
            window = rasterio.windows.Window(px, py, 1, 1)
            data = src.read(window=window, masked=True)

            # Return the first pixel value
            return data[0, 0, 0]  # Assuming you're interested in the first band
        except IndexError:
            # The lon/lat is not within the bounds of the GeoTIFF
            return None

config = OmegaConf.load('../config/surfrad.yaml')
# lon = config.stations.PSU.Longitude
# lat = config.stations.PSU.Latitude
lon = -97.710
lat = 30.338

# Define the source CRS (WGS84)
source_crs = "EPSG:4326"

# Define the target CRS (the CRS of your GeoTIFF file)
target_crs = "EPSG:32614"  # Replace this with the actual CRS of your GeoTIFF file

# Create a transformer object
transformer = Transformer.from_crs(source_crs, target_crs)

# Convert the longitude and latitude
lon, lat = transformer.transform(lon, lat)

# Example usage
geotiff_path = '/home/yuhaoliu/Code/UrbanSurfTemp/data/Austin/lst/LC08_ST_B10_20170303.tif'
with rasterio.open(geotiff_path) as src:
    print(src.crs)
pixel_value = query_geotiff(lon, lat, geotiff_path)
print("Pixel value:", pixel_value)
