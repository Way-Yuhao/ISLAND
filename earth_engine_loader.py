import os
import os.path as p
from abc import ABC
import ee
import geemap
import cv2
import torch
import torchvision
from torchvision import transforms as T
from torchvision.datasets.vision import VisionDataset
import numpy as np
import glob
from config import *


class EarthEngineLoader(VisionDataset):

    def __init__(self, root: str, geemap_obj, bounding_box, image_meta, label_meta):
        super().__init__(root)
        self.map = geemap_obj
        self.boundary = bounding_box
        self.image_meta, self.label_meta = image_meta, label_meta
        self.crop_height_coord, self.crop_width_coord = CROP_HEIGHT_COORD, CROP_WIDTH_COORD  # degrees in longitude

        self.sample_tile = ee.Image(self.image_meta['id']).select(self.image_meta['selected_bands'])
        label_dataset = ee.ImageCollection(self.label_meta['collection_id'])
        self.label_tile = label_dataset.filter(ee.Filter.eq('system:index', self.label_meta['filter'])) \
            .first().select(self.label_meta['selected_bands'])

    def __len__(self):
        # approximated number of non-overlapping samples in the boundary
        boundary = np.array(self.boundary)[0]
        # print(boundary)
        w_min, w_max = boundary[:, 0].min(), boundary[:, 0].max()
        h_min, h_max = boundary[:, 1].min(), boundary[:, 1].max()
        w_num_samples = (w_max - w_min) / CROP_WIDTH_COORD
        h_num_samples = (h_max - h_min) / CROP_HEIGHT_COORD
        return int(w_num_samples * h_num_samples)

    def __repr__(self):
        return f"{self.image_meta['name']} with bands {self.image_meta['selected_bands']}"

    def __getitem__(self, item):
        sample_region = self.get_random_sample_region()
        input_sample, label_sample = self.get_numpy(sample_region)
        # pre-processing input
        input_sample = self.scale(input_sample)
        input_sample = torch.from_numpy(input_sample).permute(2, 0, 1)
        # pre-processing label
        label_sample = self.modify_labels_super(label_sample)
        label_sample = torch.from_numpy(label_sample)  # .unsqueeze(0)

        return input_sample, label_sample

    def get_random_sample_region(self):
        boundary = np.array(self.boundary)[0]
        # print(boundary)
        w_min, w_max = boundary[:, 0].min(), boundary[:, 0].max()
        h_min, h_max = boundary[:, 1].min(), boundary[:, 1].max()

        # draw random point as upper_left corner
        w = np.random.uniform(w_min, w_max - self.crop_width_coord)  # corrected
        h = np.random.uniform(h_min, h_max - self.crop_height_coord)
        sample_coord = [[[w, h], [w, h + self.crop_height_coord],
                         [w + self.crop_width_coord, h + self.crop_height_coord], [w + self.crop_width_coord, h]]]
        sample_polygon = ee.Geometry.Polygon(sample_coord)
        sample_region = ee.Geometry(sample_polygon, None, False)
        return sample_region

    def get_numpy(self, sample_region):
        """

        :param sample_region:
        :return: numpy array of shape (h, w, c)
        """
        # TODO: consider changing to float32
        image_sample_np = geemap.ee_to_numpy(self.sample_tile, region=sample_region, default_value=0).astype('float32')
        label_sample_np = geemap.ee_to_numpy(self.label_tile, region=sample_region, default_value=0).astype('float32')
        image_sample_np = cv2.resize(image_sample_np, (256, 256), interpolation=cv2.INTER_LINEAR)
        label_sample_np = cv2.resize(label_sample_np, (256, 256), interpolation=cv2.INTER_NEAREST)
        label_sample_np = np.rint(label_sample_np).astype('int8')
        return image_sample_np, label_sample_np

    def scale(self, sample):
        """
        scale pixel values in each band using the 'scale' and 'offset' values provided by Earth Engine Data Catalog
        :param sample:
        :return:
        """
        scales = np.array(self.image_meta['scales'])
        offsets = np.array(self.image_meta['offsets'])
        return sample * scales + offsets

    def modify_labels_super(self, l):
        """
        Changes label pixel values to 4 super classes:
        * 0: developed surfaces
        * 1: water-related surfaces
        * 2: vegetated surfaces
        * 3: barren land
        :param l: unmodified label sample
        :return: modified label sample
        """
        l[(l >= 20) & (l <= 29)] = 1  # developed surfaces
        l[(l == 30) | (l == 31)] = 2  # barren land
        l[(l >= 40) & (l <= 49)] = 3  # vegetated surfaces
        l[(l >= 52) & (l <= 89)] = 4  # low-density vegetation
        l[((l >= 10) & (l <= 12)) | ((l >= 90) & (l <= 95))] = 5  # water surfaces

        assert l.min() >= VIS_PARAM['label_min'] and l.max() <= VIS_PARAM['label_max'], \
            f"ERROR: unexpected label values. Got min = {l.min()} and max = {l.max()}"
        return l

    def augment(self):
        raise NotImplementedError


class SeqEarthEngineLoader(EarthEngineLoader):
    """
    Samples a given region using raster-scanning, from lower-left corner scanning row by row.
    """
    def __init__(self, root: str, geemap_obj, bounding_box, image_meta, label_meta):
        super().__init__(root, geemap_obj, bounding_box, image_meta, label_meta)
        self.w_idx, self.h_idx = 0, 0

    def __len__(self):
        # approximated number of non-overlapping samples in the boundary
        boundary = np.array(self.boundary)[0]
        # print(boundary)
        w_min, w_max = boundary[:, 0].min(), boundary[:, 0].max()
        h_min, h_max = boundary[:, 1].min(), boundary[:, 1].max()
        self.w_num_samples = int(np.floor((w_max - w_min) / CROP_WIDTH_COORD))
        self.h_num_samples = int(np.floor((h_max - h_min) / CROP_HEIGHT_COORD))
        return int(self.w_num_samples * self.h_num_samples)

    def get_seq_sample_region(self):
        boundary = np.array(self.boundary)[0]
        w_min, w_max = boundary[:, 0].min(), boundary[:, 0].max()
        h_min, h_max = boundary[:, 1].min(), boundary[:, 1].max()
        w = w_min + CROP_WIDTH_COORD * self.w_idx
        h = h_min + CROP_HEIGHT_COORD * self.h_idx
        assert w + CROP_WIDTH_COORD < w_max, "ERROR: crop width exceeded boundary"
        assert h + CROP_HEIGHT_COORD < h_max, "ERROR: crop height exceeded boundary"
        sample_coord = [[[w, h], [w, h + self.crop_height_coord],
                         [w + self.crop_width_coord, h + self.crop_height_coord], [w + self.crop_width_coord, h]]]
        sample_polygon = ee.Geometry.Polygon(sample_coord)
        sample_region = ee.Geometry(sample_polygon, None, False)
        return sample_region

    def __getitem__(self, item):
        """

        :param item:
        :return: a scaled input numpy array, and a unmodified label numpy array
        """
        sample_region = self.get_seq_sample_region()
        input_sample, label_sample = self.get_numpy(sample_region)

        # pre-processing input
        input_sample = self.scale(input_sample)
        # input_sample = torch.from_numpy(input_sample).permute(2, 0, 1)

        # pre-processing label
        # label_sample = self.modify_labels_super(label_sample)
        # label_sample = torch.from_numpy(label_sample)

        # update indices
        if self.w_idx < self.w_num_samples - 1:
            self.w_idx += 1
        else:
            self.w_idx = 0
            self.h_idx += 1

        return input_sample, label_sample


class LocalCropLoader(VisionDataset):

    def __init__(self, root: str):
        super().__init__(root)
        self.input_dir = p.join(self.root, 'input')
        self.label_dir = p.join(self.root, 'label')
        assert (len(os.listdir(self.input_dir)) == len(os.listdir(self.label_dir))), \
            f"ERROR: number of inputs {len(os.listdir(self.input_dir))} " \
            f"does not match the number of labels {len(os.listdir(self.label_dir))}"

        self.cloud_upper_bound = LANDSAT8_META['cloud_dist_bound']
        self.m = LANDSAT8_META['selected_bands'].index('ST_CDIST') # channel index for cloud mask

    def __len__(self):
        return len(glob.glob1(self.input_dir, '*.npy'))

    def __repr__(self):
        return f"Local crop data loader at {self.root} of size {len(self)}"

    def __getitem__(self, item):
        input_sample = np.load(p.join(self.input_dir, f'{item}.npy'))
        label_sample = np.load(p.join(self.label_dir, f'{item}.npy'))
        input_sample = input_sample.squeeze(0) if len(input_sample.shape) == 4 else input_sample
        label_sample = label_sample.squeeze(0) if len(label_sample.shape) == 3 else label_sample
        # pre-process input cloud mask
        input_sample = self.pre_process_cloud_mask(input_sample)
        # pre-processing label
        label_sample = self.modify_labels_super(label_sample, input_sample)

        input_sample = torch.from_numpy(input_sample).permute(2, 0, 1)
        label_sample = torch.from_numpy(label_sample)
        return input_sample, label_sample

    def pre_process_cloud_mask(self, hs_input):
        """

        :param hs_input: numpy ndarray of shape (h, w, c)
        :return:
        """
        mask = hs_input[ :, :, self.m].copy()
        mask[mask > self.cloud_upper_bound] = 1
        hs_input[ :, :, self.m] = mask
        return hs_input

    def modify_labels_super(self, l, hs_input):
        """
        Changes label pixel values to 4 super classes:
        * 0: developed surfaces
        * 1: water-related surfaces
        * 2: vegetated surfaces
        * 3: barren land
        :param l: unmodified label sample
        :return: modified label sample
        """
        l[(l >= 20) & (l <= 29)] = 1  # developed surfaces
        l[(l == 30) | (l == 31)] = 2  # barren land
        l[(l >= 40) & (l <= 49)] = 3  # vegetated surfaces
        l[(l >= 52) & (l <= 89)] = 4  # low-density vegetation
        l[((l >= 10) & (l <= 12)) | ((l >= 90) & (l <= 95))] = 5  # water surfaces

        assert l.min() >= VIS_PARAM['label_min'] and l.max() <= VIS_PARAM['label_max'], \
            f"ERROR: unexpected label values. Got min = {l.min()} and max = {l.max()}"

        # mask out clouds
        mask = hs_input[ :, :, self.m].copy()
        l[mask == 0] = 0
        return l

    def augment(self):
        raise NotImplementedError