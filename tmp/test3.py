import os
import ee
from datetime import datetime, date, timedelta
from omegaconf import DictConfig, OmegaConf
from util.ee_utils import (acquire_reference_date, generate_cycles,
                           get_landsat_lst, get_landsat_capture_time, load_ee_image,
                           is_landsat_pixel_clear)
from eval.surfrad import get_surfrad_surf_temp_at
import pandas as pd
from rich.progress import Progress

start_date = '20140101'
end_date = '20200101'
scene_id = '016032'
config = OmegaConf.load

config = OmegaConf.load('../config/surfrad.yaml')
lon = config.stations.PSU.Longitude
lat = config.stations.PSU.Latitude
ref_date = acquire_reference_date(start_date, scene_id)
print('ref date =', ref_date)
cycles = generate_cycles(ref_date, end_date)
# print(cycles)
data = []
with Progress() as progress:
    task_id = progress.add_task("[cyan]Processing...", total=len(cycles))
    for date_ in cycles:
        progress.update(task_id, advance=1)
        try:
            image = load_ee_image(f'LANDSAT/LC08/C02/T1_L2/LC08_016032_{date_}')
        except ee.EEException as e:
            continue
        landsat_lst = get_landsat_lst(lon, lat, image=image)
        capture_time = get_landsat_capture_time(image=image)
        surfrad_lst = get_surfrad_surf_temp_at('PSU', capture_time)
        condition_clear = is_landsat_pixel_clear(lon, lat, image=image)
        data.append({
            'date': date_,
            'landsat_lst': landsat_lst,
            'surfrad_lst': surfrad_lst,
            'condition_clear': condition_clear,
            'delta_lst': landsat_lst - surfrad_lst
        })
df = pd.DataFrame(data)
print(df)

# date_ = acquire_reference_date('20130101', '016032')
# print(date_)