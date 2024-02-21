import os
import os.path as p
import pandas as pd
from abc import ABC, abstractmethod
import cv2
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import textwrap
from config.config import *
from util.filters import *
from util.helper import deprecated, rprint, yprint


class BaseInterpolator(ABC):
    """
    Abstract class for interpolators.
    """

    def __init__(self, root, target_date=None, no_log=False):
        self.root = root  # root directory
        self.cloud_path = p.join(root, 'cloud')
        self.cloud_shadow_path = p.join(root, 'cloud_shadow')
        self.output_path = p.join(root, 'output')
        if not p.exists(self.output_path):
            os.mkdir(self.output_path)
        self.target = None  # ground truth image, without synthetic occlusion
        self.target_valid_mask = None  # true valid mask, constrained by data loss
        self.target_date = target_date
        # self.get_target(target_date)  # child class specific
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
        # child class attributes
        self.interp_mode = None  # to be defined in child class
        return

    # TODO check mode for all function calls
    def get_frame(self, frame_date, mode):
        """
        Loads an image corresponding to a specified date.
        :param frame_date:
        :return: ndarray for the target bt image
        """
        if mode in ['bt', 'bt_series']:
            mode = 'bt_series'
        elif mode == 'lst':
            pass
        elif mode == 'cloud':
            pass
        elif mode == 'shadow':
            pass
        else:
            raise ValueError(f'Unexpected mode encountered. Got {mode}')
        parent_dir = p.join(self.root, mode)
        img_files = os.listdir(parent_dir)
        target_file = [f for f in img_files if frame_date in f and 'nlcd' not in f]
        target_file = [f for f in target_file if 'aux' not in f]  # files generated by geemap visualization
        if len(target_file) == 1:
            frame = cv2.imread(p.join(parent_dir, target_file[0]), -1)
        elif len(target_file) == 0:
            raise FileNotFoundError(f'Target date {frame_date} does not exist in {parent_dir}')
        else:
            raise FileExistsError(
                f'Multiple ({len(target_file)}) files found for target date {frame_date} in {parent_dir}')
        # clean up
        frame[np.isnan(frame)] = -1
        frame[np.isinf(frame)] = -1
        if np.any(frame == -1):
            pass
        return frame

    def get_target(self, target_date):
        """
        get target bt image and build valid mask. Pixel values of valid mask is false where there is a registration
        error or there is a cloud/cloud shadow for the ground truth image. Notice that such cloud/cloud shadow mask is
        imperfect.
        :param target_date:
        :return:
        """
        self.target = self.get_frame(frame_date=target_date, mode=self.interp_mode)
        # clean up target (invalid pixels due to registration)
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
            temp_img = self.target.copy()
            cloud_img = self.get_frame(frame_date=self.target_date, mode='cloud')
            shadow_img = self.get_frame(frame_date=self.target_date, mode='shadow')
        else:
            temp_img = self.get_frame(frame_date=alt_date, mode=self.interp_mode)
            cloud_img = self.get_frame(frame_date=alt_date, mode='cloud')
            shadow_img = self.get_frame(frame_date=alt_date, mode='shadow')
        valid_mask = cloud_img + shadow_img
        valid_mask = ~np.array(valid_mask, dtype=np.bool_)
        valid_mask[temp_img < 0] = False
        return valid_mask

    def get_nlcd(self):
        """
        Load a pre-aligned NLCD land cover map corresponding to a target LANDSAT temperature map
        :return:
        """
        files = os.listdir(self.root)
        nlcds = [f for f in files if 'nlcd' in f and '.tif' in f]
        nlcds = [f for f in nlcds if '._' not in f]
        nlcd_rgb_path = p.join(self.root, [f for f in nlcds if 'color' in f][0])
        nlcd_path = p.join(self.root, [f for f in nlcds if 'color' not in f][0])
        nlcd = cv2.imread(nlcd_path, -1)
        nlcd_rgb = cv2.imread(nlcd_rgb_path, -1)
        nlcd_rgb = cv2.cvtColor(nlcd_rgb, cv2.COLOR_BGR2RGB)
        assert nlcd is not None and nlcd_rgb is not None
        return nlcd, nlcd_rgb

    def correct_nlcd_key_error(self, to_key=11):
        """
        Arbitrarily assign pixels with values 0 with a specified key.
        These pixels are usually in marine regions and are presumed to be open water
        :param to_key: default is 11, open water
        :return:
        """
        nlcd_ = self.nlcd.copy()
        unlabeled_pixels = nlcd_ == 0
        error_pixel_count = np.count_nonzero(unlabeled_pixels)
        if error_pixel_count > 0:
            yprint(f'{error_pixel_count} pixels NLCD pixels are not labeled. Replacing with {to_key}')
            nlcd_[nlcd_ == 0] = to_key
            self.nlcd = nlcd_
        return

    def _clean(self, img, mask=None):
        """
        Replace pixel values with 0 for all locations where valid mask is False. By default, it uses
        target_valid_mask attribute. Masks image in place AND returns masked image.
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

    def display(self, img, error_cbar=False, msg='', xlabel_text=None):
        if type(img) is np.ndarray:
            pass
        elif img in ['gt', 'target', 't']:
            img = self.target
            msg = 'Ground Truth'
        elif img in ['occluded', 'o']:
            img = self.occluded_target
            msg = 'Occluded'
        elif img in ['reconst', 'r']:
            img = self.reconstructed_target
            msg = 'Reconstructed'
        else:
            raise AttributeError('Unknown image to display specified. Must be either a reserved string like gt, '
                                 'occluded, reconst, error, or an np.ndarray object.')
        assert img is not None
        if error_cbar:
            max_, min_ = 15, -15
            cmap_ = 'seismic'
        else:
            min_ = img[img > 250].min()
            max_ = min(330, img.max())
            cmap_ = 'magma'
        plt.imshow(img, cmap=cmap_, vmin=min_, vmax=max_)
        plt.xlabel(xlabel_text)
        plt.title(f'{msg} Temperature on {self.target_date}')
        plt.colorbar(label='Kelvin')
        plt.show()
        return

    @deprecated  # reason: clean
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

        plt.colorbar(label='Kelvin')
        plt.show()
        return 0

    def add_occlusion(self, fpath=None, use_true_cloud=False):
        """
        Adds synthetic occlusion according to a bitmap. The occluded region will have pixel values of 0
        :param fpath: path to the occlusion bitmap file. A synthetic occlusion bitmask will be generated,
        where occluded regions will have pixel values of True (1).
        :return: fractions of pixels being occluded
        """
        if fpath is not None and use_true_cloud is False:
            assert p.exists(fpath), f'{fpath} does not exist.'
            self.synthetic_occlusion = cv2.imread(fpath, -1)
            self.synthetic_occlusion = np.array(self.synthetic_occlusion, dtype=np.bool_)  # wrap in binary form
            assert (self.synthetic_occlusion is not None)
            occlusion = self.synthetic_occlusion
            self.occlusion_id = fpath[-12:-4]
        elif fpath is None and use_true_cloud is True:
            if self.target_valid_mask is None:
                self.target_valid_mask = self.build_valid_mask()
            occlusion = ~self.target_valid_mask
            self.synthetic_occlusion = occlusion  # FIXME
            self.occlusion_id = self.target_date
        else:
            raise AttributeError()
        self.occluded_target = self.target.copy()
        self.occluded_target[occlusion] = 0
        px_count = occlusion.shape[0] * occlusion.shape[1]
        occlusion_percentage = np.count_nonzero(occlusion) / px_count
        # print(f"{occlusion_percentage:.3%} of pixels added arbitrary occlusion"
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

    def save_output(self, msg=''):
        """
        Saves a NumPy array (unscaled) and a PyPlot (scaled for visualization) file for reconstruction
        result. Requires reconstruction result to exist.
        :return: None
        """
        if msg != '':
            msg = f'_{msg}'
        if self.reconstructed_target is None:
            raise ValueError('Reconstruction result does not exist. Cannot save image output')
        else:
            try:
                img = self.reconstructed_target
                # save numpy array
                output_filename = f'reconst_t{self.target_date}_syn{self.occlusion_id}_ref{self.ref_frame_date}{msg}'
                np.save(p.join(self.output_path, output_filename),
                        img)  # float32 recommended. float16 only saves 1 decimal

                # save pyplot
                # min_ = img[img > 250].min()
                # max_ = min(330, img.max())

                min_ = 270
                max_ = 330

                cmap_ = 'magma'
                plt.imshow(img, cmap=cmap_, vmin=min_, vmax=max_)
                plt.title(f'Reconstructed Brightness Temperature on {self.target_date}')
                plt.colorbar(label='BT(Kelvin)')
                output_filename = f'reconst_t{self.target_date}_syn{self.occlusion_id}_ref{self.ref_frame_date}{msg}.png'
                plt.savefig(p.join(self.output_path, output_filename))
                print('Pyplot vis saved to ', output_filename)
            except ValueError as e:
                rprint(f'ERROR: {e}.\n Current image not saved.')
            plt.close()
        return

    ######################### interpolation schemes  #########################

    @abstractmethod
    def run_interpolation(self):
        return

    def calc_avg_temp_for_class(self, c: int):
        assert str(c) in NLCD_2019_META['lut'], "ERROR: invalid NLCD class"
        temp_for_c = self.target.copy()
        temp_for_c[self.nlcd != c] = 0

        px_count = np.count_nonzero(temp_for_c)
        avg_temp = np.sum(temp_for_c) / px_count

        min_ = self.target[self.target > 250].min()
        max_ = min(330, self.target.max())
        plt.imshow(temp_for_c, cmap='magma', vmin=min_, vmax=max_)
        plt.title(f'Temperature on {self.target_date} for class {c}')
        plt.colorbar(label='Kelvin')
        plt.show()

        return avg_temp, px_count

    def calc_temp_per_class(self):
        overall_p_count = np.count_nonzero(self.target)
        for c, _ in NLCD_2019_META['lut'].items():
            t, p_count = self.calc_avg_temp_for_class(c=int(c))
            print(f'cLass {c} | average temp = {t:.2f} | freq = {p_count * 100 / overall_p_count: .2f}%')

    def plot_violins(self, show=True, include_class_agnostic=False):
        plt.figure(figsize=(16, 5))
        df = pd.DataFrame({'class': [], 'bt': []})
        palette = []
        i = 0
        for c, _ in NLCD_2019_META['lut'].items():
            c = int(c)
            temp_for_c = self.target.copy()
            temp_for_c[self.nlcd != c] = 0
            dp = temp_for_c[np.where(temp_for_c != 0)]
            if len(dp > 0):
                x = len(dp) * [NLCD_2019_META['class_names'][str(c)] + f'\n({np.var(dp):.2f})']
                new_df = pd.DataFrame({'class': x, 'bt': dp})
                df = pd.concat([df, new_df], ignore_index=True)
                palette += ['#' + NLCD_2019_META['lut'][str(c)]]
                print(f'class = {x[0]}, var = {np.var(dp):.2f}')
            i += 1
        if include_class_agnostic:
            dp = self.target[np.where(self.target != 0)]
            x = len(dp) * ['All classes' + f'\n({np.var(dp):.2f})']
            new_df = pd.DataFrame({'class': x, 'bt': dp})
            df = pd.concat([df, new_df], ignore_index=True)
            palette += ['#FFFFFF']
            print(f'class = all, var = {np.var(dp):.2f}')
        ax = sns.violinplot(x='class', y='bt', data=df, palette=palette)
        ax.set_xticklabels(textwrap.fill(x.get_text(), 11) for x in ax.get_xticklabels())
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('NLCD Land Cover Class', fontsize=18)
        plt.ylabel('Temperature (K)', fontsize=18)
        plt.tight_layout()
        if show:
            plt.show()

    def add_random_occlusion(self, size, num_occlusions):
        assert size > 0
        assert num_occlusions > 0

        max_iterations = 1000
        min_area_threshold = .9
        single_occlusion_px_size = size ** 2

        max_h = self.target.shape[0] - size
        max_w = self.target.shape[1] - size

        if self.target_valid_mask is None:
            self.target_valid_mask = self.build_valid_mask()
        real_occlusion = ~self.target_valid_mask.copy()
        all_occlusion_mask = real_occlusion.copy()  # np.bool
        i, occlusions_added = 0, 0
        while True:
            new_occlusion = np.zeros_like(all_occlusion_mask)  # np.bool
            h = np.random.randint(0, max_h)
            w = np.random.randint(0, max_w)
            new_occlusion[h: h + size, w: w + size] = True
            occlusion_be_to_added = np.logical_and(new_occlusion, ~all_occlusion_mask)  # union
            # print(np.count_nonzero(occlusion_be_to_added))
            if np.count_nonzero(occlusion_be_to_added) > single_occlusion_px_size * min_area_threshold:
                all_occlusion_mask += new_occlusion  # only add occlusion if overlap smaller than threshold
                occlusions_added += 1
            i += 1
            if occlusions_added >= num_occlusions:
                break
            if i >= max_iterations:
                print(f'max number of iterations reached. Only {occlusions_added} out of {num_occlusions} have been'
                      f' added.')
                break
        occlusion_synthetic_only = np.logical_and(all_occlusion_mask, ~real_occlusion)
        pxs_occluded_synthetic = np.count_nonzero(occlusion_synthetic_only)
        print(
            f'{pxs_occluded_synthetic} ({pxs_occluded_synthetic / (real_occlusion.shape[0] * real_occlusion.shape[1] / 100):.3f}%)'
            f' pixels are artificially occluded')
        self.synthetic_occlusion = all_occlusion_mask.copy()
        self.occluded_target = self.target.copy()
        self.occluded_target[all_occlusion_mask] = 0
        return occlusion_synthetic_only

    def add_existing_occlusion(self, occlusion_path):
        """
        To be used in evaluation mode only. Adds an existing randomly generated occlusion bitmap. Such occlusion mask
        must be generated via self.add_occlusion_mask().
        :param occlusion_path:
        :return:
        """
        if self.target_valid_mask is None:
            self.target_valid_mask = self.build_valid_mask()
        real_occlusion = ~self.target_valid_mask.copy()
        all_occlusion_mask = real_occlusion.copy()  # np.bool

        occlusion_synthetic_only = np.load(occlusion_path)
        assert occlusion_synthetic_only is not None
        pxs_occluded_synthetic = np.count_nonzero(occlusion_synthetic_only)
        print(
            f'{pxs_occluded_synthetic} ({pxs_occluded_synthetic / (real_occlusion.shape[0] * real_occlusion.shape[1] / 100):.3f}%)'
            f' pixels are artificially occluded')
        all_occlusion_mask += occlusion_synthetic_only
        self.synthetic_occlusion = all_occlusion_mask.copy()
        self.occluded_target = self.target.copy()
        self.occluded_target[all_occlusion_mask] = 0
        return occlusion_synthetic_only

    def calc_loss_hybrid(self, metric, synthetic_only_mask):
        """
        Calculate loss while expecting the reconstruction contains hybrid occlusions,
        both real and synthetic
        :param metric: mae, mse, rmse, mape
        :param synthetic_only_mask:
        :return: loss and if mae loss is specified, error_map
        """
        if self.reconstructed_target is None:
            raise AttributeError('Reconstruction image does not exist')
        a, b = self.target, self.reconstructed_target
        if metric == "mae":
            error_map = np.abs(a - b)
        elif metric == 'mse':
            error_map = np.square(a - b)
        elif metric == 'rmse':
            error_map = np.square(a - b)
        elif metric == 'mape':
            error_map = np.abs(a - b)
            error_map = error_map / a
        else:
            raise AttributeError(f'Unknown loss function encountered: {metric}.')

        error_map[~synthetic_only_mask] = 0
        loss = np.sum(error_map) / np.count_nonzero(synthetic_only_mask)
        if metric == 'rmse':
            loss = np.sqrt(loss)
        if metric != 'mae':
            error_map = None
        return loss, error_map

    def save_timelapse_frame(self, suffix=''):
        suffix = f'_{suffix}' if suffix != '' else ''
        if self.reconstructed_target is None:
            raise ValueError('Reconstruction result does not exist. Cannot save image output')
        try:
            # save image output
            img = self.reconstructed_target.copy()
            output_filename = f'reconst_{self.target_date}{suffix}'
            np.save(p.join(self.output_path, output_filename), img)
            output_vmin = 270
            output_vmax = 330
            plt.imshow(img, cmap='magma', vmax=output_vmax, vmin=output_vmin)
            plt.title(f'Reconstructed Temperature on {self.target_date}')
            plt.colorbar(label='Kelvin')
            output_filename = f'reconst_{self.target_date}{suffix}.png'
            plt.savefig(p.join(self.output_path, output_filename))
        except ValueError as e:
            rprint(f'ERROR: {e}.\n Current image not saved.')
        plt.close()
        return

    def save_error_frame(self, mask, suffix=''):
        suffix = f'_{suffix}' if suffix != '' else ''
        try:
            # save error map
            img = self.target - self.reconstructed_target
            img[~mask] = 0
            output_filename = f'error_{self.target_date}{suffix}'
            np.save(p.join(self.output_path, output_filename), img)
            error_vmin = -5
            error_vmax = 5
            plt.imshow(img, cmap='seismic', vmax=error_vmax, vmin=error_vmin)
            plt.title(f'Error in Temperature on {self.target_date} under synthetic occlusions only')
            plt.colorbar(label='Kelvin')
            output_filename = f'error_{self.target_date}{suffix}.png'
            plt.savefig(p.join(self.output_path, output_filename))
        except ValueError as e:
            rprint(f'ERROR: {e}.\n Current error map not saved.')
        plt.close()

