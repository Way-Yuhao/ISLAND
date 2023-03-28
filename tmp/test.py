from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from matplotlib import pyplot as plt


def project_npy():
    gdf = gpd.GeoDataFrame()
    geoms = []
    dim = 1
    coords = [[[-95.690165, 29.5937], [-95.690165, 30.266005], [-94.900379, 30.266005], [-94.900379, 29.5937]]]
    coords = coords[0]
    for ln, lt in coords:
        geom = Polygon([(ln, lt), ((ln + dim), lt), ((ln + dim), (lt - dim)), (ln, (lt - dim))])
        geoms.append(geom)
    gdf['geometry'] = geoms
    print(gdf)


def project_npy2():
    coords = [[[-95.690165, 29.5937], [-95.690165, 30.266005], [-94.900379, 30.266005], [-94.900379, 29.5937]]]
    coords = coords[0]
    # img = np.load('../data/Houston/output_st/npy/st_20170116.npy')
    img = np.load('../data/Houston/analysis/hotspots_aggregate.npy')
    # print(img)
    h, w = img.shape
    tl = GroundControlPoint(0, 0, coords[1][0], coords[1][1])
    bl = GroundControlPoint(h, 0, coords[0][0], coords[0][1])
    br = GroundControlPoint(h, w, coords[3][0], coords[3][1])
    tr = GroundControlPoint(0, w, coords[2][0], coords[2][1])
    gcps = [tl, bl, br, tr]
    transform = from_gcps(gcps)
    crs = 'epsg:4326'

    new_dataset = rio.open('../data/Houston/analysis/hotspots_aggregate_geotiff.tif', 'w', driver='GTiff', height=h, width=w, count=1,
                           dtype=str(img.dtype),
                           crs=crs,
                           # crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                           transform=transform)
    new_dataset.write(img, 1)
    new_dataset.close()


def check_geotiff():
    src = rio.open('../data/Houston/analysis/hotspots_aggregate_geotiff.tif')
    arr = src.read(1)
    plt.imshow(arr)
    plt.show()


def main():
    # project_npy2()
    # check_geotiff()


if __name__ == '__main__':
    main()
