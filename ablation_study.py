"""
Conduct ablation study with synthetic occlusion for a specified city.
The following model outputs will be generated:
* Full + spatial only + temporal only -> .../output_eval_full
* Minus NLCD -> .../output_no_nlcd
"""
__author__ = 'yuhao liu'

import os
import os.path as p
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from interpolator import Interpolator
from util.helper import rprint, yprint, parse_csv_dates
from batch_eval import timelapse_with_synthetic_occlusion


def move_output_to(city_name, to_dir):
    origin_dir = f'./data/{city_name}/output'
    dest_dir = f'./data/{city_name}/{to_dir}'
    assert p.exists(origin_dir)
    assert not p.exists(dest_dir)
    os.rename(origin_dir, dest_dir)
    yprint(f'Evaluation output from full model moved to {dest_dir}.')
    return


def run_no_nlcd(city_name, dates, resume=False):
    """
    Requres full model output already stored at designated directory
    :param city_name:
    :return:
    """
    root_ = f'./data/{city_name}'
    log_fpath = f"./data/{city_name}/output/timelapse.csv"
    full_output_dir = f'./data/{city_name}/output_eval_full'
    assert p.exists(full_output_dir)
    log = []
    for d in dates:
        if resume:
            existing_output_files = os.listdir(p.join(root_, 'output'))
            current_date_files = [f for f in existing_output_files if d in f]
            if len(current_date_files) > 0:
                print(f'Found outputs for date {d}. Skipped.')
                continue
        yprint(f'Evaluating {d} without NLCD')
        interp = Interpolator(root=root_, target_date=d)
        # read existing synthetic occlusion for this date
        # TODO: after testing, saving syn occlusion again is no longer required.
        syn_occlusion_path = p.join(full_output_dir, f'syn_occlusion_{d}.npy')
        added_occlusion = interp.add_existing_occlusion(syn_occlusion_path)
        output_filename = f'syn_occlusion_{d}'
        np.save(p.join(interp.output_path, output_filename), added_occlusion)
        plt.imshow(added_occlusion)
        plt.title(f'Added synthetic occlusion on {d}')
        output_filename = f'syn_occlusion_{d}.png'
        plt.savefig(p.join(interp.output_path, output_filename))
        interp.ablation_interp_no_nlcd()  # run ablation method
        loss, error_map = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
        interp.save_error_frame(mask=added_occlusion, suffix='st')
        print(f'MAE loss over synthetic occluded areas = {loss:.3f}')
        log += [(d, loss, np.count_nonzero(added_occlusion))]
    df = pd.DataFrame(log, columns=['target_date', 'mae', 'synthetic occlusion percentage'])
    df.to_csv(log_fpath, index=False)
    print('csv file saved to ', log_fpath)


def ablation(city_name):
    dates = parse_csv_dates(city_name)
    if p.exists(f'./data/{city_name}/output/'):
        raise FileExistsError('Output directory already exists. Please move the files elsewhere by '
                              'renaming the folder.')
    # timelapse_with_synthetic_occlusion('Houston', resume=False)
    # move_output_to(city_name, to_dir='output_eval_full')
    run_no_nlcd(city_name, dates, resume=False)
    move_output_to(city_name, to_dir='output_eval_no_nlcd')


def main():
    ablation('Houston')


if __name__ == '__main__':
    main()
