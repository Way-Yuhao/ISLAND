import os
import os.path as p
import cv2
import numpy as np
import pandas as pd
import random
from util.helper import rprint, yprint, hash_, pjoin, save_cmap, get_season, deprecated


class OcclusionSampler:
    """
    Samples occlusion from a list of cities, after data augmentation
    Requires
    * an averages_by_date.csv file stored in ../data/[CITY]/analysis
    """
    def __init__(self, data_dir, city_list):
        self.data_dir = data_dir
        self.city_list = city_list
        self.df = self.build_df()

    def build_df(self):
        df = pd.DataFrame()
        for city in self.city_list:
            # log_path = f'./data/{city}/analysis/averages_by_date.csv'
            log_path = pjoin(self.data_dir, city, 'analysis', 'averages_by_date.csv')
            if not p.exists(log_path):
                rprint(f'File for {city} does not exist')
                continue
            current_df = pd.read_csv(log_path)
            current_df['city'] = city
            df = pd.concat([df, current_df], ignore_index=True)
        df['range'] = df.apply(lambda row: self.categorize(row), axis=1)
        return df

    def sample(self, theta_range, shape):
        """
        Samples an occlusion of type bool, applies random data augmentation,
        and reshapes to desired shape.
        :param theta_range:
        :param shape:
        :return:
        """
        rows = self.df[self.df['range'] == theta_range]
        n = len(rows.index)  # number of rows
        k = random.randint(0, n-1)
        row = rows.iloc[k]
        city, d = row['city'], row['date']
        # cloud = cv2.imread(f'./data/{city}/cloud/LC08_cloud_{d}.tif', -1)
        # shadow = cv2.imread(f'./data/{city}/shadow/LC08_shadow_{d}.tif', -1)
        cloud = cv2.imread(pjoin(self.data_dir, city, 'cloud', f'LC08_cloud_{d}.tif'), -1)
        shadow = cv2.imread(pjoin(self.data_dir, city, 'shadow', f'LC08_shadow_{d}.tif'), -1)
        cirrus = cv2.imread(pjoin(self.data_dir, city, 'cirrus', f'LC08_cirrus_{d}.tif'), -1)
        occlusion = cloud + shadow + cirrus
        occlusion[occlusion != 0] = 255
        occlusion = occlusion.astype(np.float32)
        occlusion = self.augment(occlusion)
        occlusion = self.resize_(occlusion, shape)
        return occlusion

    def augment(self, img):
        rand_code = random.randint(0, 2)
        img = self.flip(img, rand_code)
        return img

    @staticmethod
    def flip(img, code):
        if code == 0 or code == 1:  # flip horizontally or vertically
            pass
        elif code == 2:  # flip horizontally and vertically
            code = [0, 1]
        return np.flip(img, code)

    @staticmethod
    def categorize(row):
        theta = row['theta']
        if theta < 0.9:
            cat = int(theta * 10) / 10  # take floor
        elif theta < 0.99:
            cat = 0.9
        else:
            cat = 1.0
        return cat

    def resize_(self, img, shape):
        img = cv2.resize(img, dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
        return img.astype(bool)
