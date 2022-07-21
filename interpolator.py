"""
Interpolates the TOA Brightness Temperature (B10 band from Landsat 8/9)
"""
import os
import os.path as p
import numpy as np
import pandas as pd
from tqdm import tqdm
import abc
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap
from config import *
from util.filters import *
from util.helper import deprecated, rprint, yprint


class Interpolator(abc.ABC):

    def __init__(self, root, target_date=None):
        self.root = root  # root directory
        self.bt_path = p.join(root, 'bt_series')
        self.cloud_path = p.join(root, 'cloud')
        self.cloud_shadow_path = p.join(root, 'cloud_shadow')
        self.output_path = p.join(root, 'output')
        if not p.exists(self.output_path):
            os.mkdir(self.output_path)
        self.target = None  # ground truth image, without synthetic occlusion
        self.target_valid_mask = None  # true valid mask, constrained by data loss
        self.target_date = target_date
        self.get_target(target_date)
        self.nlcd, self.nlcd_rgb = self.get_nlcd()
        self.synthetic_occlusion = None  # artificially introduced occlusion
        self.occlusion_id = None  # id for synthetic occlusion
        self.occluded_target = None
        self.reconstructed_target = None
        self._num_classes = len(NLCD_2019_META['lut'].items())  # number of NLCD classes, including those absent
        return

    def get_frame(self, target_date, mode='bt'):
        """
        Loads an image corresponding to a specified date.
        :param target_date:
        :return: ndarray for the target bt image
        """
        if mode in ['bt', 'bt_series']:
            mode = 'bt_series'
        elif mode == 'cloud':
            pass
        elif mode == 'shadow':
            pass
        else:
            raise ValueError(f'Unexpected mode encountered. Got {mode}')
        parent_dir = p.join(self.root, mode)
        img_files = os.listdir(parent_dir)
        target_file = [f for f in img_files if target_date in f and 'nlcd' not in f]
        if len(target_file) == 1:
            target = cv2.imread(p.join(parent_dir, target_file[0]), -1)
        elif len(target_file) == 0:
            raise FileNotFoundError(f'Target date {target_date} does not exist in {parent_dir}')
        else:
            raise FileExistsError(
                f'Multiple ({len(target_file)}) files found for target date {target_date} in {parent_dir}')
        # clean up
        target[np.isnan(target)] = -1
        target[np.isinf(target)] = -1
        if np.any(target == -1):
            print(f"{bcolors.WARNING}{target_file} contains np.nan or np.NIF, which has been converted to -1"
                  f"{bcolors.ENDC}")
        return target

    def get_target(self, target_date):
        """
        get target bt image and build valid mask. Pixel values of valid mask is false where there is a registration
        error or there is a cloud/cloud shadow for the ground truth image. Notice that such cloud/cloud shadow mask is
        imperfect.
        :param target_date:
        :return:
        """
        self.target = self.get_frame(target_date=target_date)
        # clean up target (invalid pixels due to registration)
        # self.target_valid_mask = np.ones_like(self.target, dtype=np.bool_)
        # self.target_valid_mask[self.target < 0] = False
        self.target_valid_mask = self.build_valid_mask()
        self.target[self.target < 0] = 0  # overwrite np.nan or -inf with 0

    def build_valid_mask(self, alt_date=None):
        """
        computers a binary bitmask representing the validity of pixel values for a BT map on a given day. Pixels marked
        as True are valid pixels. Valid pixels satisfy (1) no cloud, and (2) no cloud shadow, and (3) bt reading greater
        than 0 K.
        :param alt_date:
        :return:
        """
        if alt_date is None:  # using default target date
            bt_img = self.target.copy()
            cloud_img = self.get_frame(target_date=self.target_date, mode='cloud')
            shadow_img = self.get_frame(target_date=self.target_date, mode='shadow')
        else:
            bt_img = self.get_frame(target_date=alt_date, mode='bt_series')
            cloud_img = self.get_frame(target_date=alt_date, mode='cloud')
            shadow_img = self.get_frame(target_date=alt_date, mode='shadow')
        valid_mask = cloud_img + shadow_img
        valid_mask = ~np.array(valid_mask, dtype=np.bool_)
        valid_mask[bt_img < 0] = False
        return valid_mask

    def get_nlcd(self):
        """
        Load a pre-aligned NLCD landcover map corresponding to a target LANDSAT temperature map
        :return:
        """
        nlcd = cv2.imread(p.join(self.root, 'nlcd_houston_20180103.tif'), -1)
        nlcd_rgb = cv2.imread(p.join(self.root, 'nlcd_houston_color.tif'), -1)
        nlcd_rgb = cv2.cvtColor(nlcd_rgb, cv2.COLOR_BGR2RGB)
        return nlcd, nlcd_rgb

    def _clean(self, img, mask=None):
        """
        Replace pixel values with 0 for all locations where valid mask is False. By default, it uses
        target_valid_mask attribute.
        :param img:
        :return:
        """
        if mask is None:
            if self.target_valid_mask is None:
                raise AttributeError
            else:
                mask = self.target_valid_mask
        img[~mask] = 0
        return img

    def _clear_outputs(self):
        self.reconstructed_target = None

    def display_target(self, mode=None, img=None, text=None):
        """
        Displays a plot via matplotlib for the desired matrix
        :param mode:
        :param img:
        :param text:
        :return:
        """
        # plt.figure(figsize=(20, 20))
        if mode == 'gt' and img is None:
            img = self.target
            msg = 'Ground Truth'
        elif mode == 'occluded' and img is None:
            img = self.occluded_target
            assert self.occluded_target is None
            msg = 'Occluded'
        elif mode == 'reconst' and img is None:
            img = self.reconstructed_target
            assert self.reconstructed_target is not None
            msg = 'Reconstructed'
        elif mode == 'error' and img is None:
            if self.reconstructed_target is None:
                img = self._clean(self.occluded_target - self.target)
                msg = 'Error (occluded)'
            else:
                img = self._clean(self.reconstructed_target - self.target)
                msg = 'Error (reconstructed)'
        elif img is not None:  # img to display is included in args
            if img.max() == img.min():
                print('Empty image invoked to display. Skipped.')
                return -1
            msg = 'Custom'
        else:
            raise AttributeError('Unknown display mode. Choose among {gt, occluded, reconst}')
        # matplotlib.use('macosx')
        if mode != 'error':
            min_ = img[img > 250].min()
            max_ = min(330, img.max())
            cmap_ = 'magma'
        else:
            # max_delta = max(img.max(), -img.min())
            # max_ = max_delta
            # min_ = -max_delta
            max_, min_ = 15, -15  # FIXME
            cmap_ = 'seismic'
        plt.imshow(img, cmap=cmap_, vmin=min_, vmax=max_)
        plt.xlabel(text)
        plt.title(f'{msg} Brightness Temperature on {self.target_date}')

        plt.colorbar(label='BT(Kelvin)')
        plt.show()
        return 0

    def add_occlusion(self, fpath):
        """
        Adds synthetic occlusion according to a bitmap. The occluded region will have pixel values of 0
        :param fpath: path to the occlusion bitmap file. A synthetic occlusion bitmask will be generated,
        where occluded regions will have pixel values of True (1).
        :return: fractions of pixels being occluded
        """
        assert p.exists(fpath)
        self.synthetic_occlusion = cv2.imread(fpath, -1)
        self.synthetic_occlusion = np.array(self.synthetic_occlusion, dtype=np.bool_)  # wrap in binary form
        assert (self.synthetic_occlusion is not None)

        self.occluded_target = self.target.copy()
        self.occluded_target[self.synthetic_occlusion] = 0
        px_count = self.synthetic_occlusion.shape[0] * self.synthetic_occlusion.shape[1]
        occlusion_percentage = np.count_nonzero(self.synthetic_occlusion) / px_count
        # print(f"{occlusion_percentage:.3%} of pixels added arbitrary occlusion")
        self.occlusion_id = fpath[-12:-4]
        return occlusion_percentage

    def calc_loss(self, metric='mae', print_=False, entire_canvas=False):
        """
        calculates the mean absolute error (MAE) over the synthetically occluded area
        :return:
        """

        if self.reconstructed_target is None:
            print('No reconstruction map found. Calculating loss on initial occluded image')
            a, b = self.target, self.occluded_target
        else:
            a, b = self.target, self.reconstructed_target

        if metric == "mae":
            error_map = np.abs(a - b)
        elif metric == 'mse':
            error_map = np.square(a - b)
        else:
            raise AttributeError(f'Unknown loss function encountered: {metric}. ')
        error_map = self._clean(error_map)
        if not entire_canvas:  # by default, only calculate loss on synthetic occluded regions
            loss = np.sum(error_map) / np.count_nonzero(self.synthetic_occlusion)
        else:  # calculate loss on entire canvas
            loss = np.average(error_map)


        if print_:
            print(f'{metric} loss = {loss:.3f}')
        return loss

    def clear_outputs(self):
        self.reconstructed_target = None
        return

    def save_output(self):
        """
        Saves a NumPy array (unscaled) and a PyPlot (scaled for visualization) file for reconstruction
        result. Requires reconstruction result to exist.
        :return: None
        """

        if self.reconstructed_target is None:
            raise ValueError('Reconstruction result does not exist. Cannot save image output')
        else:
            img = self.reconstructed_target
            # save numpy array
            output_filename = f'r_{self.target_date}_{self.occlusion_id}'
            np.save(p.join(self.output_path, output_filename), img)  # float32 recommended. float16 only saves 1 decimal

            # save pyplot
            min_ = img[img > 250].min()
            max_ = min(330, img.max())
            cmap_ = 'magma'
            plt.imshow(img, cmap=cmap_, vmin=min_, vmax=max_)
            plt.title(f'Reconstructed Brightness Temperature on {self.target_date}')
            plt.colorbar(label='BT(Kelvin)')
            output_filename = f'r_{self.target_date}_{self.occlusion_id}.png'
            plt.savefig(p.join(self.output_path, output_filename))
            print('Pyplot vis saved to ', output_filename)
            plt.close()
        return

    def run_interpolation(self):
        self.spatial_interp()

    def spatial_interp(self, f=None):
        self._nlm_local(f)
        # self._nlm_global()

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

    def _nlm_global(self):
        """
        spatial channel, global rectangular filter. May throw Value Error when there does not exist any
        replacement candidate for some class in the entire canvas.
        :return:
        """
        print(f"SPATIAL FILTER: global filter")
        self.reconstructed_target = self.occluded_target

        px_count = np.count_nonzero(self.occluded_target)
        default_avg_temp = np.sum(self.occluded_target) / px_count  # global, class-agnostic

        for c, _ in NLCD_2019_META['lut'].items():
            c = int(c)
            temp_for_c = self.occluded_target.copy()
            temp_for_c[self.nlcd != c] = 0  # remove other classes

            # show map for each class
            # plt.imshow(temp_for_c)
            # plt.title(c)
            # plt.show()

            px_count = np.count_nonzero(temp_for_c)
            avg_temp = np.sum(temp_for_c) / px_count if px_count else None

            # pixels corresponding to current class and requiring filling values
            replacement_bitmap = np.zeros_like(self.occluded_target, dtype=np.bool_)
            replacement_bitmap[self.nlcd == c] = True
            replacement_bitmap[self.synthetic_occlusion == 0] = False

            if avg_temp is not None and np.any(replacement_bitmap):  # requires in-paint, data available
                self.reconstructed_target[replacement_bitmap] = avg_temp
            elif avg_temp is None and np.any(replacement_bitmap):  # requires in-paint, data unavailable
                # raise ValueError(f'Unable to acquire average temperature for class {c}')
                yprint(f'Unable to acquire average temperature for class {c}. Defaulting to global average.')
                self.reconstructed_target[replacement_bitmap] = default_avg_temp

    def _nlm_local(self, f=100):
        """
        spatial channel, local gaussian filter with kernel size of f
        :param f: kernel size, in pixels
        :return:
        """
        assert f is not None, "filter size cannot be None"
        print(f"SPATIAL FILTER: local gaussian filter with f={f}")
        self.reconstructed_target = self.occluded_target
        x_length, y_length = self.occluded_target.shape

        temp_class_c = {}  # temperature map for each class
        # average temperature for all pixels under each class
        avg_temp = {}
        for c, _ in NLCD_2019_META['lut'].items():
            c = int(c)
            cur = self.occluded_target.copy()
            cur[self.nlcd != c] = 0  # remove other classes
            temp_class_c[c] = cur
            px_counts = np.count_nonzero(temp_class_c[c])
            avg_temp[c] = np.sum(temp_class_c[c]) / px_counts if px_counts != 0 else None

        print("processing missing pixels...")
        no_local_data_counter = 0
        pbar = tqdm(total=np.count_nonzero(self.synthetic_occlusion))
        x_cord, y_cord = np.nonzero(self.synthetic_occlusion)
        for x, y in zip(x_cord, y_cord):
            c = self.nlcd[x, y]
            # rectangular kernel
            x_left, x_right = max(0, x - f), min(x_length - 1, x + f)
            y_left, y_right = max(0, y - f), min(y_length - 1, y + f)
            local_region = temp_class_c[c][x_left: x_right + 1, y_left: y_right + 1]
            local_region_bin = np.array(local_region, dtype=np.bool_)
            if np.any(local_region_bin):  # data available in local region
                kernel = gkern(canvas=local_region.shape, center=(x - x_left, y - y_left), sig=f / 2)
                kernel[~local_region_bin] = 0
                est_temp = np.sum(local_region * kernel) / np.sum(kernel)
                self.reconstructed_target[x, y] = est_temp
            else:  # data unavailable in local region
                self.reconstructed_target[x, y] = avg_temp[c]
                no_local_data_counter += 1
            pbar.update()
        pbar.close()
        print(f'{no_local_data_counter} pixels ({no_local_data_counter / (x_length * y_length):.5%}) '
              f'used global calculations')

    def temporal_interp_as_is(self, ref_frame_date):
        """
        obtain a BT frame from the past as-is. No adjustment or cloud masking applied.
        :param ref_frame_date:
        :return:
        """
        past_frame = self.get_frame(ref_frame_date)
        # need to filter out clouds
        self.reconstructed_target = past_frame
        return

    def temporal_interp_global_adj(self, ref_frame_date):
        """
        obtain a BT frame from the past, adjusted using global mean (class agnostic). No cloud masking.
        :param ref_frame_date:
        :return:
        """
        target_frame = self.target.copy()
        past_frame = self.get_frame(ref_frame_date)
        # need to filter out clouds
        reconst_img = past_frame + (target_frame.mean() - past_frame.mean())
        self.reconstructed_target = reconst_img
        return

    def temporal_interp(self, ref_frame_date):
        # load one image from the past
        past_frame = self.get_frame(ref_frame_date)
        target_frame = self.occluded_target.copy()
        reconst_img = np.zeros_like(target_frame, dtype=np.float32)
        target_avgs, past_avgs = {}, {}  # mean temperature (scalar) for all pixels in each class
        for c, _ in NLCD_2019_META['lut'].items():
            c = int(c)
            past_c = past_frame.copy()
            target_c = target_frame.copy()
            past_c[self.nlcd != c] = 0
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
                compensated_past_c = past_c.copy()
                compensated_past_c[compensated_past_c != 0] += target_avgs[c] - past_avgs[c]
                reconst_img += compensated_past_c
            else:
                # raise AttributeError(c)  # TODO
                pass
        self.reconstructed_target = reconst_img

    def temporal_interp_cloud(self, ref_frame_date, ref_syn_cloud_date):
        """
        Requires target frame
        :param ref_frame_date:
        :param ref_syn_cloud_date:
        :return:
        """
        # TODO: add cloud masking
        target_frame = self.occluded_target.copy()
        # target_frame = self._clean(target_frame)
        # load one image from the past

        past_interp = Interpolator(root=self.root, target_date=ref_frame_date)
        past_interp.add_occlusion(fpath=p.join(past_interp.root, 'cloud',
                                               f'LC08_cloud_houston_{ref_syn_cloud_date}.tif'))
        past_interp._nlm_global()
        past_frame = past_interp.reconstructed_target  # complete

        # past_interp.display_target(mode='gt')


        # past_frame = self.get_frame(ref_frame_date)
        # past_mask = self.get_frame(ref_syn_cloud_date, mode='cloud').astype(np.bool_)
        # past_mask = self.build_valid_mask(alt_date=ref_syn_cloud_date)
        # past_frame = self._clean(img=past_frame, mask=past_mask)  # TODO: check this

        reconst_img = np.zeros_like(target_frame, dtype=np.float32)
        target_avgs, past_avgs = {}, {}  # mean temperature (scalar) for all pixels in each class
        for c, _ in NLCD_2019_META['lut'].items():
            c = int(c)
            past_c = past_frame.copy()
            target_c = target_frame.copy()
            past_c[self.nlcd != c] = 0
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
                compensated_past_c = past_c.copy()
                compensated_past_c[compensated_past_c != 0] += target_avgs[c] - past_avgs[c]
                reconst_img += compensated_past_c
            else:
                # raise AttributeError(c)  # TODO
                pass
        reconst_img = self._clean(reconst_img)
        self.reconstructed_target = reconst_img

    def heat_cluster_interp(self):
        raise NotImplementedError

    def calc_avg_temp_for_class(self, c: int):
        assert str(c) in NLCD_2019_META['lut'], "ERROR: invalid NLCD class"
        temp_for_c = self.target.copy()
        temp_for_c[self.nlcd != c] = 0

        px_count = np.count_nonzero(temp_for_c)
        avg_temp = np.sum(temp_for_c) / px_count

        min_ = self.target[self.target > 250].min()
        max_ = min(330, self.target.max())
        plt.imshow(temp_for_c, cmap='magma', vmin=min_, vmax=max_)
        plt.title(f'Brightness Temperature on {self.target_date} for class {c}')
        plt.colorbar(label='BT(Kelvin)')
        plt.show()

        return avg_temp, px_count

    def calc_temp_per_class(self):
        overall_p_count = np.count_nonzero(self.target)
        for c, _ in NLCD_2019_META['lut'].items():
            t, p_count = self.calc_avg_temp_for_class(c=int(c))
            print(f'cLass {c} | average temp = {t:.2f} | freq = {p_count * 100 / overall_p_count: .2f}%')

    @deprecated
    def plot_scatter_class(self):
        plt.figure(figsize=(10, 5))
        i = 0
        for c, _ in NLCD_2019_META['lut'].items():
            c = int(c)
            temp_for_c = self.target.copy()
            temp_for_c[self.nlcd != c] = 0
            dp = temp_for_c[np.where(temp_for_c != 0)]
            if len(dp > 0):
                x = np.ones_like(dp) * i
                y_mean = np.average(dp)
                y_std = np.std(dp)
                plt.scatter(x=x, y=dp, s=3, c='#' + NLCD_2019_META['lut'][str(c)],
                            label=NLCD_2019_META['class_names'][str(c)])
                plt.errorbar(x=i, y=y_mean, yerr=y_std, fmt='.', color='black', capsize=3)
                i += 1
        plt.xlabel('NLCD Landcover Class')
        plt.ylabel('Brightness Temperature (K)')
        plt.xticks([])
        plt.title('Distribution of Brightness Temperature per Landcover Class')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=5)
        plt.tight_layout()
        plt.show()

    def plot_violins(self):
        plt.figure(figsize=(15, 5))
        sns.set(style='whitegrid')
        df = pd.DataFrame({'class': [], 'bt': []})
        palette = []
        i = 0
        for c, _ in NLCD_2019_META['lut'].items():
            c = int(c)
            temp_for_c = self.target.copy()
            temp_for_c[self.nlcd != c] = 0
            dp = temp_for_c[np.where(temp_for_c != 0)]
            if len(dp > 0):
                x = len(dp) * [NLCD_2019_META['class_names'][str(c)]]
                new_df = pd.DataFrame({'class': x, 'bt': dp})
                df = pd.concat([df, new_df], ignore_index=True)
                palette += ['#' + NLCD_2019_META['lut'][str(c)]]
            i += 1
        ax = sns.violinplot(x='class', y='bt', data=df, palette=palette)
        ax.set_xticklabels(textwrap.fill(x.get_text(), 11) for x in ax.get_xticklabels())
        plt.xlabel('NLCD Landcover Class')
        plt.ylabel('Brightness Temperature (K)')
        plt.title('Distribution of Brightness Temperature per Landcover Class')
        plt.tight_layout()
        plt.show()
