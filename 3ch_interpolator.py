"""
Interpolates the TOA Brightness Temperature (B10 band from Landsat 8/9)
"""
import time
import datetime
import os
import os.path as p
from multiprocessing import Manager, Pool
import numpy as np
from tqdm import tqdm
import abc
import cv2
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from config import *
from util.filters import *


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
        if target_date is not None:
            self.get_target(target_date)
        self.target_date = target_date
        self.nlcd, self.nlcd_rgb = self.get_nlcd()
        self.synthetic_occlusion = None  # artificially introduced occlusion
        self.occlusion_id = None  # id for synthetic occlusion
        self.occluded_target = None
        self.reconstructed_target = None
        self.anim = None  # matplotlib animation
        return

    def get_target(self, target_date):
        """
        get target bt image and build valid mask. Pixel values of valid mask is false where there is a registration
        error or there is a cloud/cloud shadow for the ground truth image. Notice that such cloud/cloud shadow mask is
        imperfect.
        :param target_date:
        :return:
        """
        parent_dir = p.join(self.root, 'bt_series')
        bt_files = os.listdir(parent_dir)
        target_file = [f for f in bt_files if target_date in f]
        if len(target_file) == 1:
            self.target = cv2.imread(p.join(parent_dir, target_file[0]), -1)
        elif len(target_file) == 0:
            raise FileNotFoundError(f'Target date {target_date} does not exist in {parent_dir}')
        else:
            raise FileExistsError(
                f'Multiple ({len(target_file)})files found for target date {target_date} in {parent_dir}')

        # clean up target (invalid pixels due to registration)
        self.target_valid_mask = np.ones_like(self.target, dtype=np.bool_)
        self.target_valid_mask[self.target < 0] = False
        self.target[self.target < 0] = 0  # overwrite np.nan or -inf with 0

    def get_nlcd(self):
        """
        Load a pre-aligned NLCD landcover map corresponding to a target LANDSAT temperature map
        :return:
        """
        nlcd = cv2.imread(p.join(self.root, 'nlcd_houston_20180103.tif'), -1)
        nlcd_rgb = cv2.imread(p.join(self.root, 'nlcd_houston_color.tif'), -1)
        nlcd_rgb = cv2.cvtColor(nlcd_rgb, cv2.COLOR_BGR2RGB)
        return nlcd, nlcd_rgb

    def _clean(self, img):
        """
        Replace pixel values with 0 for all locations where valid mask is False
        :param img:
        :return:
        """
        if self.target_valid_mask is None:
            raise AttributeError
        img[~self.target_valid_mask] = 0
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

        if mode == 'gt':
            img = self.target
            msg = 'Ground Truth'
        elif mode == 'occluded':
            img = self.occluded_target
            assert self.occluded_target is not None
            msg = 'Occluded'
        elif mode == 'reconst':
            img = self.reconstructed_target
            assert self.reconstructed_target is not None
            msg = 'Reconstructed'
        elif mode == 'error':
            if self.reconstructed_target is None:
                img = self._clean(self.occluded_target - self.target)
                msg = 'Error (occluded)'
            else:
                img = self._clean(self.reconstructed_target - self.target)
                msg = 'Error (reconstructed)'
        elif mode is None and img is not None:
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
            max_delta = max(img.max(), -img.min())
            max_ = max_delta
            min_ = -max_delta
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

    def calc_loss(self, metric='mae', print_=False):
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
        loss = np.sum(error_map) / np.count_nonzero(self.synthetic_occlusion)

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

    def run_interpolation(self):
        self.spatial_interp()

    def spatial_interp(self, f=None):
        self._nlm_local(f)
        # self._nlm_global()

    def _nlm_global(self):
        """
        spatial channel, global rectangular filter. May throw Value Error when there does not exist any
        replacement candidate for some class in the entire canvas.
        :return:
        """
        print(f"SPATIAL FILTER: global filter")
        self.reconstructed_target = self.occluded_target
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
                raise ValueError(f'Unable to acquire average temperature for class {c}')
            # self.display_target(mode='reconst')

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

    def temporal_interp(self):
        raise NotImplementedError

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


def evaluate_multiprocess(num_procs=4):
    start_time = time.monotonic()
    stats_fpath = './data/spatial_intperp_series_mp.csv'
    interp = Interpolator(root='./data/export/', target_date='20181221')
    dataset_path = os.listdir(p.join(interp.root, 'cloud'))
    for f in dataset_path:  # clean up irrelevant input files
        if f[-3:] != 'tif' or f[:4] == 'nlcd':
            dataset_path.remove(f)
    print(f"Evaluating {len(dataset_path)} scenes")

    pool = Pool(num_procs)
    manager = Manager()
    lock = manager.Lock()
    r = None  # return values
    # empty lists, compatible with multi-processes
    cloud_percs, maes, mses = manager.list(), manager.list(), manager.list()
    for f in dataset_path:
        relative_path = p.join(interp.root, 'cloud', f)
        r = pool.apply_async(eval_single, args=(relative_path, lock, cloud_percs, maes, mses))
    r.get()
    pool.close()
    pool.join()
    print('---------------------------------')
    print(print(f"{bcolors.WARNING}May have encountered error. Scroll up to view."))
    d = {'cloud_perc': list(cloud_percs), 'MAE': list(maes), 'MSE': list(mses)}
    df = pd.DataFrame(data=d)
    df.to_csv(stats_fpath)
    print("CSV file saved to ", stats_fpath)
    stop_time = time.monotonic()
    print('Processing time = ', datetime.timedelta(seconds=stop_time - start_time))


def eval_single(occlusion_fpath, lock, cloud_percs, maes, mses):
    interp = Interpolator(root='./data/export/', target_date='20181221')
    cloud_perc = interp.add_occlusion(occlusion_fpath)
    try:
        # interp.fill_average()
        interp.spatial_interp(f=100)

        mae = interp.calc_loss(print_=False, metric='mae')
        mse = interp.calc_loss(print_=False, metric='mse')
        print(f"{cloud_perc:.3%} | mae = {mae:.3f} | mse = {mse:.3f}")
        interp.save_output()
    except ValueError as e:
        print(f"{bcolors.FAIL}ERROR: {e}{bcolors.ENDC}")

    with lock:  # to ensure atomic IO operations
        cloud_percs.append(cloud_perc)
        maes.append(mae)
        mses.append(mse)
    del interp
    return


def evaluate():
    start_time = time.monotonic()
    stats_fpath = './data/spatial_intperp_series.csv'
    interp = Interpolator(root='./data/export/', target_date='20181221')
    dataset_path = os.listdir(p.join(interp.root, 'cloud'))
    print(f"Evaluating {len(dataset_path)} scenes")
    cloud_percs, maes, mses = [], [], []
    for f in dataset_path:
        if f[-3:] != 'tif' or f[:4] == 'nlcd':
            continue
        cloud_perc = interp.add_occlusion(p.join(interp.root, 'cloud', f))

        try:
            interp.spatial_interp(f=100)
            # interp.fill_average()
        except ValueError:
            pass
        mae = interp.calc_loss(print_=False, metric='mae')
        mse = interp.calc_loss(print_=False, metric='mse')
        print(f"{cloud_perc:.3%} | mae = {mae:.3f} | mse = {mse:.3f}")
        cloud_percs.append(cloud_perc)
        interp.save_output()
        maes.append(mae)
        mses.append(mse)

    print('---------------------------------')
    d = {'cloud_perc': cloud_percs, 'MAE': maes, 'MSE': mses}
    df = pd.DataFrame(data=d)
    df.to_csv(stats_fpath)
    print("CSV file saved to ", stats_fpath)
    stop_time = time.monotonic()
    print('Processing time = ', datetime.timedelta(seconds=stop_time - start_time))


def main():
    interp = Interpolator(root='./data/export/', target_date='20181221')
    # fpath = p.join(interp.root, 'cirrus', 'LC08_cirrus_houston_20181018.tif')
    # fpath = p.join(interp.root, 'cirrus', 'LC08_cirrus_houston_20190903.tif')
    fpath = p.join(interp.root, 'cirrus', 'LC08_cirrus_houston_20190311.tif')
    interp.add_occlusion(fpath)
    # interp.fill_average()
    # interp.display_target(mode='occluded')
    # interp.calc_loss(print_=True)
    # t = interp.calc_avg_temp_for_class(c=11)
    # print(t)
    # interp.calc_temp_per_class()
    # interp.plot_scatter_class()

    # mean = np.mean(interp.target)
    # mean_img = np.ones_like(interp.target) * mean
    # interp.reconstructed_target = mean_img
    # interp.calc_loss(print_=True, metric='mae')
    # interp.calc_loss(print_=True, metric='mse')

    interp.spatial_interp()
    interp.calc_loss(print_=True)
    interp.display_target(mode='error')
    interp.display_target(mode='reconst')


if __name__ == '__main__':
    # main()
    # evaluate()
    evaluate_multiprocess(num_procs=10)
