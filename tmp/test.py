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

    # exporting to google drive
    # geemap.ee_export_image_to_drive(img, description="entire_image_all_bands", folder="export", region=None, scale=30)
    # geemap.ee_export_image_to_drive(img.select('ST_B10'), description="entire_image_bt10", folder="export", region=None, scale=30)
    # geemap.ee_export_image_to_drive(img.select('ST_B10').multiply(0.00341802).add(149), description="entire_image_scaled", folder="export", region=None, scale=30)
    # geemap.ee_export_image_to_drive(img.select('ST_B10').multiply(0.00341802).add(149), description="cropped_scaled", folder="export", region=str(export_boundary), scale=30)


if __name__ == '__main__':
    download()
