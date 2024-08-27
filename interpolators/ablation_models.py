__author__ = 'yuhao liu'
import os
import os.path as p
import numpy as np
import cv2
from interpolators.interpolator import BaseInterpolator
from interpolators.lst_interpolator import LST_Interpolator
from config.config import *


class LST_Interp_No_NLCD(LST_Interpolator):

    def __init__(self, root, target_date=None, no_log=False):
        super().__init__(root, target_date, no_log)

    def get_nlcd(self):
        files = os.listdir(self.root)
        nlcds = [f for f in files if 'nlcd' in f and '.tif' in f]
        nlcds = [f for f in nlcds if '._' not in f]
        nlcd_path = p.join(self.root, [f for f in nlcds if 'color' not in f][0])
        nlcd = cv2.imread(nlcd_path, -1)
        nlcd = np.ones_like(nlcd) * 11
        nlcd_rgb = nlcd.copy()
        return nlcd, nlcd_rgb

    def temporal_interp(self, ref_frame_date, spatial_kern_size, global_threshold=.5):
        """
        performs temporal interpolation with respect to one specified reference frame. This method does not introduce
        synthetic occlusion to reference frame.
        :param global_threshold: if the cloud coverage percentage of reference frame is above this threshold,
               then use global filter
        :param ref_frame_date:
        :return: real occlusion percentage for the reference frame
        """
        # load one image from the past
        target_frame = self.occluded_target.copy()
        reconst_img = np.zeros_like(target_frame, dtype=np.float32)
        target_avgs, past_avgs = {}, {}  # mean temperature (scalar) for all pixels in each class

        self.ref_frame_date = ref_frame_date
        ref_frame = self.get_frame(self.ref_frame_date, mode=self.interp_mode)

        ref_interp = LST_Interp_No_NLCD(root=self.root, target_date=self.ref_frame_date)
        ref_occlusion_percentage = ref_interp.add_occlusion(use_true_cloud=True)
        if ref_occlusion_percentage <= global_threshold:  # use local gaussian
            ref_interp._nlm_local(f=spatial_kern_size)
            if np.isnan(ref_interp.reconstructed_target).any():
                ref_interp.reconstructed_target = None
                ref_interp._nlm_global()
        else:  # use global rectangular
            ref_interp._nlm_global()
        complete_ref_frame = ref_interp.reconstructed_target.copy()  # pre-processed ref frame, spatially complete

        for c, _ in NLCD_2019_META['lut'].items():
            c = int(c)
            past_c = ref_frame.copy()
            target_c = target_frame.copy()
            complete_past_c = complete_ref_frame.copy()
            past_c[self.nlcd != c] = 0
            complete_past_c[self.nlcd != c] = 0
            target_c[self.nlcd != c] = 0

            target_avg_pixels = target_c[np.where(target_c != 0)]
            target_avg_pixels = target_avg_pixels[target_avg_pixels >= 0]  # clean up invalid pixels
            past_avg_pixels = past_c[np.where(past_c != 0)]
            past_avg_pixels = past_avg_pixels[past_avg_pixels >= 0]  # clean up invalid pixels
            if len(target_avg_pixels) != 0:
                target_avgs[c] = np.average(target_avg_pixels)
            if len(past_avg_pixels) != 0:
                past_avgs[c] = np.average(past_avg_pixels)

            # build reconstruction image class by class
            if c in target_avgs and c in past_avgs:
                # compensated_past_c = past_c.copy()  # with no spatial pre-processing on reference frame
                compensated_past_c = complete_past_c.copy()  # with spatial pre-processing on reference frame
                compensated_past_c[compensated_past_c != 0] += target_avgs[c] - past_avgs[c]
                reconst_img += compensated_past_c
            else:
                # raise AttributeError(c)  # TODO
                pass

        self.reconstructed_target = np.zeros_like(self.occluded_target)
        self.reconstructed_target[self.synthetic_occlusion] = reconst_img[self.synthetic_occlusion]
        self.reconstructed_target[~self.synthetic_occlusion] = self.occluded_target[~self.synthetic_occlusion]

        del ref_interp
        return ref_occlusion_percentage


class LST_Interp_Fill_Average(LST_Interpolator):

    def __init__(self, root, target_date=None, no_log=False):
        super().__init__(root, target_date, no_log)

    def fill_average(self):
        """
        Naive baseline interpolator. This is a class-agnostic interpolator, with global rectangular filter.
        Simply fills in occluded regions with the global average
        :return:
        """
        assert self.occluded_target is not None
        self.clear_outputs()
        input_bitmask = np.array(~self.synthetic_occlusion, dtype=np.bool_)
        input_bitmask[~self.target_valid_mask] = False
        if np.any(input_bitmask):
            avg = np.average(self.occluded_target[input_bitmask])
            self.reconstructed_target = self.occluded_target.copy()
            self.reconstructed_target[self.synthetic_occlusion] = avg
            print(f"Using baseline average interpolator, with avg = {avg:.2f}")
        else:
            print('ERROR: 100% of input is occluded. No average temperature can be determined.')
            self.reconstructed_target = self.occluded_target.copy()
        self.save_timelapse_frame()  # suffix='temporal')