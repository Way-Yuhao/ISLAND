import os
import os.path as p
import datetime as dt
from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
import cv2
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import textwrap
from config import *
from util.filters import *
from util.helper import deprecated, rprint, yprint


class Interpolator(ABC):
    """
    Abstract class for interpolators.
    """

    def __init__(self, root, target_date=None, no_log=False,
                 ablation_no_nlcd=False):
        self.root = root  # root directory
        # self.bt_path = p.join(root, 'bt_series')
        self.cloud_path = p.join(root, 'cloud')
        self.cloud_shadow_path = p.join(root, 'cloud_shadow')
        self.output_path = p.join(root, 'output')
        if not p.exists(self.output_path):
            os.mkdir(self.output_path)
        ######## ablation flags ############
        self.ablation_no_nlcd = ablation_no_nlcd
        ####################################
        self.target = None  # ground truth image, without synthetic occlusion
        self.target_valid_mask = None  # true valid mask, constrained by data loss
        self.target_date = target_date
        self.get_target(target_date)
        self.nlcd, self.nlcd_rgb = self.get_nlcd()
        self.correct_nlcd_key_error()
        self.synthetic_occlusion = None  # artificially introduced occlusion
        self.occlusion_id = None  # id for synthetic occlusion
        self.occluded_target = None
        self.reconstructed_target = None
        self._num_classes = len(NLCD_2019_META['lut'].items())  # number of NLCD classes, including those absent
        if not no_log:
            df = pd.read_csv(p.join(self.root, 'metadata.csv'))
            assert df is not None
            dates = df['date'].values.tolist()
            dates = [str(d) for d in dates]
            df['date'] = dates
            self.metadata = df
        # temporal
        self.ref_frame_date = None
        return

    @abstractmethod
    def interpolate(self, data):
        pass