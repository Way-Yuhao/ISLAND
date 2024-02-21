"""
Interpolates the Landsat 8 LST temperature map using a combination of spatial and temporal interpolation.
"""
import os
import os.path as p
import datetime as dt
from datetime import timedelta
import pandas as pd
from tqdm import tqdm
from config.config import *
from util.filters import *
from util.helper import deprecated, rprint, yprint
from interpolators.interpolator import BaseInterpolator


class LST_Interpolator(BaseInterpolator):

    def __init__(self, root, target_date=None, no_log=False):
        super().__init__(root, target_date, no_log)
        self.interp_mode = 'lst'
        self.get_target(target_date)
        return

    def compute_spatio_temporal_weight(self):
        """
        Return weight w to be used for spatial channel. (1 - w) will be used for temporal
        :return:
        """
        if self.synthetic_occlusion is None:
            raise AttributeError()
        px_count = self.synthetic_occlusion.shape[0] * self.synthetic_occlusion.shape[1]
        occlusion_percentage = np.count_nonzero(self.synthetic_occlusion) / px_count
        return 1 - occlusion_percentage

    def run_interpolation(self, spatial_global_cutoff=.5):
        print('Running spatial & temporal channel...')

        px_count = self.synthetic_occlusion.shape[0] * self.synthetic_occlusion.shape[1]
        occlusion_percentage = np.count_nonzero(self.synthetic_occlusion) / px_count
        print(f'occlusion percentage (real + synth) = {occlusion_percentage:.3f}')

        # TODO: local gaussian for all?
        if occlusion_percentage > .99:
            # remote these
            self.reconstructed_target = self.occluded_target.copy()
            self.save_timelapse_frame(suffix='occluded')
            self.reconstructed_target = None

            print('Encountered 100% cloudy frame. Skipped.')
            self.reconstructed_target = np.zeros_like(self.occluded_target)
            self.save_timelapse_frame(suffix='spatial')
            self.save_timelapse_frame(suffix='temporal')
            self.save_timelapse_frame(suffix='st')
        else:
            self.reconstructed_target = None
            if occlusion_percentage < .5:
                self._nlm_local(f=75)  # spatial, local gaussian
            else:
                self._nlm_global()  # spatial, global rectangular
            self.save_timelapse_frame(suffix='spatial')
            reconst_spatial = self.reconstructed_target.copy()
            assert reconst_spatial is not None

            self.reconstructed_target = None
            try:
                self.temporal_interp_multi_frame(num_frames=3, max_delta_cycle=2, max_cloud_perc=.1)
            except ArithmeticError:
                yprint('Retrying temporal reference with max_delta_cycle = 4')
                try:
                    self.temporal_interp_multi_frame(num_frames=3, max_delta_cycle=4, max_cloud_perc=.1)
                except ArithmeticError:
                    yprint('Retrying temporal reference with max_delta_cycle = 4 and max_cloud_prec = .2')
                    self.temporal_interp_multi_frame(num_frames=3, max_delta_cycle=4, max_cloud_perc=.2)
            # assume temporal computation is successful
            self.save_timelapse_frame(suffix='temporal')
            reconst_temporal = self.reconstructed_target.copy()
            assert reconst_temporal is not None

            self.reconstructed_target = None
            w_spatial = self.compute_spatio_temporal_weight()
            w_temporal = 1 - w_spatial
            # self.reconstructed_target = (reconst_spatial + reconst_temporal) / 2
            self.reconstructed_target = w_spatial * reconst_spatial + w_temporal * reconst_temporal
            self.save_timelapse_frame(suffix='st')
        return

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
        self.reconstructed_target = self.occluded_target.copy()

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
                yprint(f'Unable to acquire average temperature for class {c}. Default to global average.')
                self.reconstructed_target[replacement_bitmap] = default_avg_temp

    def _nlm_local(self, f=100):
        """
        spatial channel, local gaussian filter with kernel size of f
        :param f: kernel size, in pixels
        :return:
        """
        assert f is not None, "filter size cannot be None"
        print(f"SPATIAL FILTER: local gaussian filter with f={f}")
        self.reconstructed_target = self.occluded_target.copy()
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
        pbar = tqdm(total=np.count_nonzero(self.synthetic_occlusion), position=0)
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

    def temporal_interp_multi_frame(self, num_frames, max_delta_cycle, max_cloud_perc):
        assert num_frames in range(1, 11)  # between 1 and 11 frames
        print(f'Looking for at most {num_frames} reference frame candidates subject to')
        print('\tdelta cycle < ', max_delta_cycle)
        print('\tcloud coverage percentage < ', max_cloud_perc)

        ref_dates_str = self.metadata['date'].values.tolist()
        ref_occlusion_perc = self.metadata['cloud_percentage'].values.tolist()
        ref_dates = [dt.datetime.strptime(str(d), '%Y%m%d').date() for d in ref_dates_str]
        target_date = dt.datetime.strptime(str(self.target_date), '%Y%m%d').date()

        days_delta, same_year_deltas, ref_percs = [], [], []
        for ref_date in ref_dates:
            days_delta.append(abs((target_date - ref_date).days))
            try:
                ref_same_year = ref_date.replace(year=target_date.year)
            except ValueError:  # leap year
                yprint('Encountered lead year. Using the previous day as reference.')
                ref_same_year = (ref_date - timedelta(days=1)).replace(year=target_date.year)
            same_year_delta = abs((target_date - ref_same_year).days)
            same_year_deltas.append(min(same_year_delta, 365 - same_year_delta))

        df = pd.DataFrame(ref_dates_str, columns=['ref_dates'])  # a list of all candidates for reference frames
        df['same_year_delta'] = same_year_deltas
        df['days_delta'] = days_delta
        df['ref_percs'] = ref_occlusion_perc

        df = df.loc[df['days_delta'] != 0]  # remove target frame itself
        df = df.loc[df['same_year_delta'] < max_delta_cycle * 16 + 1]  # filter by max delta cycle
        df = df.loc[df['ref_percs'] < max_cloud_perc]  # filter by max cloud coverage
        df = df.sort_values(by=['days_delta'])

        print(f'Found {len(df.index)} candidate frames that satisfy conditions:')
        if df.empty:
            yprint('No candidate reference frames satisfy conditions above. Mission aborted.')
            raise ArithmeticError()
        if num_frames < len(df.index):
            df = df.iloc[:num_frames]
        print(f'Selected {len(df.index)} frames closest to target frame in time:')
        print(df)
        selected_ref_dates = df['ref_dates'].values
        reconst_imgs = []

        for d in selected_ref_dates:
            self.temporal_interp(ref_frame_date=d)
            reconst_imgs.append(self.reconstructed_target)
            self.reconstructed_target = None

        # blended image is the average of all reconstructed images, with equal weights
        blended_img = np.zeros_like(reconst_imgs[0])
        for i in reconst_imgs:
            blended_img += i
        blended_img /= len(reconst_imgs)
        self.reconstructed_target = blended_img

    def temporal_interp(self, ref_frame_date, global_threshold=.5):
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

        ref_interp = LST_Interpolator(root=self.root, target_date=self.ref_frame_date)
        ref_occlusion_percentage = ref_interp.add_occlusion(use_true_cloud=True)
        if ref_occlusion_percentage <= global_threshold:  # use local gaussian
            ref_interp._nlm_local(f=75)
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

    @deprecated
    def temporal_interp_cloud(self, ref_frame_date, ref_syn_cloud_date):
        """
        Performs temporal interpolation after applying synthetic cloud to reference frame.
        Requires cloud-free reference frame.
        :param ref_frame_date:
        :param ref_syn_cloud_date:
        :return: synthetic occlusion percentage of the past frame
        """
        # TODO: add cloud masking
        target_frame = self.occluded_target.copy()
        # target_frame = self._clean(target_frame)
        # load one image from the past
        self.ref_frame_date = ref_frame_date
        past_interp = LST_Interpolator(root=self.root, target_date=self.ref_frame_date)
        past_syn_occlusion_perc = past_interp.add_occlusion(fpath=p.join(past_interp.root, 'cloud',
                                                                         f'LC08_cloud_{ref_syn_cloud_date}.tif'))
        past_interp._nlm_global()
        past_frame = past_interp.reconstructed_target  # complete

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
        return past_syn_occlusion_perc