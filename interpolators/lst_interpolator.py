"""
Interpolates the Landsat 8 LST temperature map using a combination of spatial and temporal interpolation.
"""
import os
import os.path as p
import datetime as dt
from datetime import timedelta
import pandas as pd
from tqdm import tqdm
from config import *
from util.filters import *
from util.helper import deprecated, rprint, yprint
from interpolators.interpolator import BaseInterpolator


class LST_Interpolator(BaseInterpolator):

    def __init__(self, root, target_date=None, no_log=False):
        super().__init__(root, target_date, no_log)
        self.interp_mode = 'lst'
        self.get_target(target_date)
        return
