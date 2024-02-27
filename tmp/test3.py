import ee
import geemap

# Initialize the Earth Engine API
ee.Initialize()

# Your specific longitude and latitude
lon = -95.43495
lat = 29.75765
point = ee.Geometry.Point(lon, lat)

# Load an image
#image = ee.Image('LANDSAT/LC08/C01/T1/LC08_044034_20140318')  # Example Landsat image
# image = ee.Image('LANDSAT/LC08/C01/T1/LC08_026047_20230124')  # Example Landsat image

dataset = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD')
# Filter the collection to the 2016 product.
image = dataset.filter(ee.Filter.eq('system:index', '2019')).first()

# Use reduceRegion to sample the image at the given point
# The reducer is set to 'first' to get the value of the pixel at the point
# scale is set according to the image resolution, adjust accordingly
info = image.reduceRegion(ee.Reducer.first(), point, scale=30).getInfo()

# Extract pixel value (example for extracting a band value)
# Replace 'B1' with the specific band you're interested in
pixel_value = info['landcover']  # Adjust 'B1' to the band you're interested in

print(pixel_value)
