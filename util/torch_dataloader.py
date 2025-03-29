from typing import List, Union, Optional, Dict
import os
import os.path as p
# import sys
import numpy as np
# from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
from natsort import natsorted
from matplotlib import pyplot as plt
from torchvision import transforms

__author__ = 'yuhao liu'

class OccludedLSTDataSet(Dataset):
    """
    Dataset for Occluded Land Surface Temperature (LST) data.
    :param dataset_dir: str, path to the root directory of the dataset.
    :param regions: Union[str, List[str]], the region(s) to be included in the dataset.
    :param filter_cp_above: Optional[float], filter out samples with cloud percentage above this threshold.
    :param use_color_nlcd: bool, whether to use color NLCD maps.

    """

    def __init__(self, dataset_dir: str, regions: Union[str, List[str]],
                 filter_cp_above: Optional[float],
                 use_color_nlcd: bool = False, *args, **kwargs):
        self.dataset_dir = dataset_dir
        self.regions = regions
        self.filter_cp_above = filter_cp_above
        self.use_color_nlcd = use_color_nlcd

        # to be defined elsewhere
        self.all_samples = None
        self.nlcd_maps = {}
        self.setup()
        return

    def setup(self):
        self.all_samples = pd.DataFrame()
        if type(self.regions) == str and self.regions.lower() == 'all':
            raise NotImplementedError()
        else:
            if isinstance(self.regions, str):
                self.regions = [self.regions]
        visited = []
        for region in self.regions:
            samples_in_region = self.scan_region(region)
            self.all_samples = pd.concat([self.all_samples, samples_in_region])
            visited.append(region)
        # print summary
        print('############# DATASET STATISTICS #############')
        print(f'Found {len(visited)} regions: {visited}')
        print(f'Total samples: {len(self.all_samples)}')
        print('##############################################')
        return

    def scan_region(self, region: str):
        samples = self.parse_metadata(region)
        samples = self.filter_by_cloud_percentage(samples)
        return samples

    def parse_metadata(self, region: str):
        metadata_file = os.path.join(self.dataset_dir, region, 'metadata.csv')
        if not p.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file {metadata_file} not found.")
        meta_data = pd.read_csv(metadata_file)
        # add a column called region
        meta_data['region'] = region
        return meta_data

    def filter_by_cloud_percentage(self, meta_data: pd.DataFrame) -> pd.DataFrame:
        if self.filter_cp_above is None:
            return meta_data
        meta_data = meta_data[meta_data['cloud_percentage'] < self.filter_cp_above]
        return meta_data

    def get_nlcd(self, region: str) -> np.ndarray:
        if region in self.nlcd_maps:
            return self.nlcd_maps[region]
        nlcd_files = os.listdir(os.path.join(self.dataset_dir, region))
        nlcd_files = [f for f in nlcd_files if 'nlcd' in f and f.endswith('.tif') and not f.startswith('.')]
        if self.use_color_nlcd:
            # nlcd_file = [f for f in nlcd_files if 'color' in f][0]
            # nlcd = cv2.imread(os.path.join(self.dataset_dir, region, nlcd_file), cv2.IMREAD_UNCHANGED)
            # nlcd = cv2.cvtColor(nlcd, cv2.COLOR_BGR2RGB)
            raise NotImplementedError('Color NLCD not supported yet.')
        else:
            nlcd_file = [f for f in nlcd_files if 'color' not in f][0]
            nlcd = cv2.imread(os.path.join(self.dataset_dir, region, nlcd_file), cv2.IMREAD_UNCHANGED)
            nlcd = torch.tensor(nlcd, dtype=torch.float32)
            # normalize to [0, 1]
            nlcd = nlcd / 100.0
        self.nlcd_maps[region] = nlcd
        return nlcd


    def build_valid_mask(self, region_: str, date_: str) -> torch.tensor:
        cloud_img = cv2.imread(os.path.join(self.dataset_dir, region_, 'cloud', f'LC08_cloud_{date_}.tif'),
                               cv2.IMREAD_UNCHANGED)
        shadow_img = cv2.imread(os.path.join(self.dataset_dir, region_, 'shadow', f'LC08_shadow_{date_}.tif'),
                                cv2.IMREAD_UNCHANGED)
        cirrus_img = cv2.imread(os.path.join(self.dataset_dir, region_, 'cirrus', f'LC08_cirrus_{date_}.tif'),
                                cv2.IMREAD_UNCHANGED)

        valid_mask = cloud_img + shadow_img + cirrus_img
        valid_mask = ~np.array(valid_mask, dtype=np.bool_)
        # valid_mask[temp_img < 0] = False # TODO: deprecated logic
        valid_mask = torch.tensor(valid_mask, dtype=torch.float32)
        return valid_mask

    def get_lst(self, region_: str, date_: str) -> torch.tensor:
        lst = cv2.imread(os.path.join(self.dataset_dir, region_, 'lst', f'LC08_ST_B10_{date_}.tif'),
                         cv2.IMREAD_UNCHANGED)
        assert lst is not None
        lst = torch.tensor(lst, dtype=torch.float32)
        # normalize to [0, 1]
        lst_max, lst_min = 250, 350 # in Kelvin
        lst = (lst - lst_min) / (lst_max - lst_min)
        return lst

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
        """
        Returns a dictionary of data items as a dictionary to torch.Tensor.
        All entries are normalized to [0, 1]
        """
        info = self.all_samples.iloc[idx]
        region_, date_ = info['region'], info['date']
        nlcd = self.get_nlcd(region_)
        valid_mask = self.build_valid_mask(region_, date_)
        lst = self.get_lst(region_, date_)
        data_dict = {'nlcd': nlcd, 'valid_mask': valid_mask, 'lst': lst}
        return data_dict

if __name__ == '__main__':
    data_root_dir = '/home/yuhaoliu/Data/ISLAND/'
    regions = ['Charlotte'] # ['Houston']
    # regions = 'all'
    a = OccludedLSTDataSet(data_root_dir, regions, filter_cp_above=0.9, use_color_nlcd=False)
    sample = a[0]
    pass