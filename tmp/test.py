import os
service_prefix = ''
os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = f"{service_prefix}/proxy/{{port}}"
import sys
import ee
import geemap
import numpy as np
import tqdm
import cv2
import seaborn as sns
from datetime import datetime
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from omegaconf import DictConfig, OmegaConf
import hydra
from eval.surfrad import get_emis_at, read_surfrad_file_from_url, get_surfrad_surf_temp_at
from util.ee_utils import get_landsat_capture_time, get_landsat_lst, cvt_lat_lon_to_path_row, query_geotiff
from config.config import NLCD_2019_META
import rasterio
from natsort import natsorted

def plot_on_map():
    vis = {
      'min': 280,
      'max': 340,
      'palette' : sns.color_palette('inferno', 20).as_hex(),
    }
    config = OmegaConf.load('../config/surfrad.yaml')
    station_id = 'BND'
    date_ = '20170611'
    lon = config['stations'][station_id]['Longitude']
    lat = config['stations'][station_id]['Latitude']
    scene_id = config['stations'][station_id]['scene_id']
    Map = geemap.Map()
    Map.add_basemap('HYBRID')

    img_path = f'/home/yuhaoliu/Code/UrbanSurfTemp/data/{station_id}/lst/LC08_ST_B10_{date_}.tif'
    # output_path = '/home/yuhaoliu/Code/UrbanSurfTemp/data/Austin/output_referenced_f75/lst/20170303.tif'
    image = ee.Image(f'LANDSAT/LC08/C02/T1_L2/LC08_{scene_id}_{date_}')
    Map.add_raster(img_path, palette=vis['palette'], vmax=vis['max'], vmin=vis['min'], layer_name='downloaded_lst')
    Map.addLayer(image.select('ST_B10').multiply(0.00341802).add(149), vis, 'landsat_lst')
    Map


def download():
    Map = geemap.Map()
    config = OmegaConf.load('../config/surfrad.yaml')
    station_id = 'PSU'
    date_ = '20170322'
    lon = config['stations'][station_id]['Longitude']
    lat = config['stations'][station_id]['Latitude']
    scene_id = config['stations'][station_id]['scene_id']
    img_path = f'LANDSAT/LC08/C02/T1_L2/LC08_{scene_id}_{date_}'
    img = ee.Image(img_path)
    filename = '/home/yuhaoliu/Code/UrbanSurfTemp/tmp/align_2.tif'
    export_boundary = config['stations'][station_id]['bounding_box']
    projection = img.projection().getInfo()

    geemap.ee_export_image(img.select('ST_B10').multiply(0.00341802).add(149),
                           filename=filename,
                           scale=30,
                           region=str(export_boundary),
                           crs=projection['crs'],
                           crs_transform=projection['transform'],
                           file_per_band=False)

def download_nlcd():
    geemap.Map()
    suffix = 'no_crs_no_scale'
    output_dir = '/home/yuhaoliu/Code/UrbanSurfTemp/tmp'

    dataset = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD')
    nlcd2019 = dataset.filter(ee.Filter.eq('system:index', '2019')).first()
    landcover_2019 = nlcd2019.select('landcover')
    filename = os.path.join(output_dir, f'nlcd_{suffix}.tif')
    export_boundary = '[[[-88.764038, 39.812459], [-87.973022, 39.812459], [-87.973022, 40.34625], [-88.764038, 40.34625], [-88.764038, 39.812459]]]'
    ref_img = ee.Image('LANDSAT/LC08/C02/T1_TOA/LC08_023032_20170203').select('B1')
    projection = ref_img.projection().getInfo()
    #
    geemap.ee_export_image(landcover_2019,
                      filename=filename,
                      scale=30,
                      region=export_boundary,
                      crs=projection['crs'],
                      # crs_transform=projection['transform'],
                      file_per_band=False)

    color_map_nlcd(filename, os.path.join(output_dir, f'nlcd_{suffix}_rgb.tif'))


def color_map_nlcd(source, dest):

    def build_nlcd_lux():
        decode_hex = lambda h: tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        rgb_lut = np.zeros((256, 3), dtype=np.uint8)
        for key, value in NLCD_2019_META['lut'].items():
            rgb_lut[int(key), :] = np.array(decode_hex(value))
        return rgb_lut

    nlcd = cv2.imread(source, -1)  #
    ref_img = rasterio.open(source)
    lut = build_nlcd_lux()
    nlcd_rgb = np.zeros_like(nlcd)
    nlcd_rgb = np.dstack((nlcd_rgb, nlcd_rgb, nlcd_rgb))
    for i in range(len(nlcd)):
        for j in range(len(nlcd[0])):
            nlcd_rgb[i, j, :] = lut[nlcd[i, j], :]

    # nlcd_rgb = cv2.cvtColor(nlcd_rgb, cv2.COLOR_BGR2RGB)
    nlcd_rgb = np.transpose(nlcd_rgb, (2, 0, 1))
    # cv2.imwrite(dest, nlcd_rgb)
    out_tif = rasterio.open(dest, 'w',
                            driver='Gtiff', height=ref_img.height, width=ref_img.width,
                            count=3, crs=ref_img.crs, transform=ref_img.transform,
                            dtype=nlcd_rgb.dtype)
    out_tif.write(nlcd_rgb)
    ref_img.close()
    out_tif.close()
    print('NLCD RGB image written to ', dest)
    return


def check_nlcd():
    # in each folder check the dimension of nlcd and one lst file match
    root = '/home/yuhaoliu/Code/UrbanSurfTemp/data'
    cities = os.listdir(root)

    # exclude specified folders
    cities = [c for c in cities if os.path.isdir(f'{root}/{c}')]

    exclude_list = ['Fort Worth', 'general', 'City3']  # replace with your actual list of cities to exclude

    # exclude specified folders
    cities = [c for c in cities if os.path.isdir(f'{root}/{c}') and c not in exclude_list]

    for city in cities:
        lst_files = natsorted(os.listdir(f'{root}/{city}/lst'))
        nlcd_files = os.listdir(f'{root}/{city}')
        nlcd_file = [f for f in nlcd_files if 'color' not in f and 'nlcd' in f][0]
        lst = cv2.imread(f'{root}/{city}/lst/{lst_files[0]}', -1)
        nlcd = cv2.imread(f'{root}/{city}/{nlcd_file}', -1)
        if lst.shape != nlcd.shape:
            print(f'Error: Dimensions of lst and nlcd images do not match for city {city}.')
            print(f'lst: {lst.shape}, nlcd: {nlcd.shape}')
        else:
            print(f'city: {city} lst: {lst.shape}, nlcd: {nlcd.shape}')
    print('All cities passed dimension check.')

def dims():
    files = natsorted(os.listdir('/home/yuhaoliu/Code/UrbanSurfTemp/data/BND/lst'))
    for f in files:
        img = cv2.imread(f'/home/yuhaoliu/Code/UrbanSurfTemp/data/BND/lst/{f}', -1)
        print(f, img.shape)

if __name__ == '__main__':
    # download_nlcd()
    # dims()
    # check_nlcd()
    print(0)
