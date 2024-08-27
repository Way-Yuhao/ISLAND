"""
Conduct ablation study with synthetic occlusion for a specified city.
The following model outputs will be generated:
* Full + spatial only + temporal only -> .../output_eval_full
* Minus NLCD -> .../output_no_nlcd
* Naive average -> .../output_naive_average
"""
__author__ = 'yuhao liu'

import os
import os.path as p
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from bt_interpolator import Interpolator
from interpolators.lst_interpolator import LST_Interpolator
from interpolators.ablation_models import LST_Interp_No_NLCD, LST_Interp_Fill_Average
from util.helper import rprint, yprint, parse_csv_dates, alert, capture_stdout
# from batch_eval import timelapse_with_synthetic_occlusion, calc_error_from_outputs
from eval_lst import timelapse_with_synthetic_occlusion, calc_error_from_outputs


def move_output_to(region_dir, to_dir):
    origin_dir = p.join(region_dir, 'output')
    dest_dir = p.join(region_dir, to_dir)
    assert p.exists(origin_dir)
    assert not p.exists(dest_dir)
    os.rename(origin_dir, dest_dir)
    yprint(f'Evaluation output moved to {dest_dir}.')
    return


def run_island(city_name, occlusion_size, num_occlusions, spatial_kern_size=75, resume=False):
    """
    Generates timelapses of BT for a given city while adding random synthetic occlusion.
    Evaluates loss only on synthetically occluded areas.
    :param resume: skip existing outputs and resume at the next frame
    :param city_name:
    :return:
    """
    root_ = '/home/yuhaoliu/Data/ISLAND/cities/{}'.format(city_name)
    log_fpath = f"/home/yuhaoliu/Data/ISLAND/cities/{city_name}/output/timelapse.csv"
    output_dir = p.join(root_, 'output')
    if not resume and p.exists(output_dir):
        raise FileExistsError(f'Output directory {output_dir}'
                              f'please either turn \'resume\' on or remove the existing '
                              f'directory.')
    df = pd.read_csv(p.join(root_, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = [str(d) for d in dates]
    log = []
    for d in dates:
        if resume:
            existing_output_files = os.listdir(p.join(root_, 'output'))
            current_date_files = [f for f in existing_output_files if d in f]
            if len(current_date_files) > 0:
                print(f'Found outputs for date {d}. Skipped.')
                continue
        yprint(f'Evaluating {d}')
        interp = LST_Interpolator(root=root_, target_date=d)
        added_occlusion = interp.add_random_occlusion(size=occlusion_size, num_occlusions=num_occlusions)
        # save added occlusion
        output_filename = f'syn_occlusion_{d}'
        np.save(p.join(interp.output_path, output_filename), added_occlusion)
        plt.imshow(added_occlusion)
        plt.title(f'Added synthetic occlusion on {d}')
        output_filename = f'syn_occlusion_{d}.png'
        plt.savefig(p.join(interp.output_path, output_filename))
        interp.run_interpolation(spatial_kern_size=spatial_kern_size)
        loss, error_map = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
        interp.save_error_frame(mask=added_occlusion, suffix='lst')
        print(f'MAE loss over synthetic occluded areas = {loss:.3f}')
        log += [(d, loss, np.count_nonzero(added_occlusion))]
    df = pd.DataFrame(log, columns=['target_date', 'mae', 'synthetic occlusion percentage'])
    df.to_csv(log_fpath, index=False)
    print('csv file saved to ', log_fpath)
    alert(f'Simulated evaluation finished for {city_name} with {num_occlusions} '
          f'synthetic occlusions of size {occlusion_size} '
          f'using spatial kernel size {spatial_kern_size}.')
    return


def run_no_nlcd(city_name, dates, resume=False):
    """
    Requres full model output already stored at designated directory
    :param city_name:
    :return:
    """
    # root_ = f'./data/{city_name}'
    # log_fpath = f"./data/{city_name}/output/timelapse.csv"
    # full_output_dir = f'./data/{city_name}/output_eval_full'

    root_ = '/home/yuhaoliu/Data/ISLAND/cities/{}'.format(city_name)
    log_fpath = f"/home/yuhaoliu/Data/ISLAND/cities/{city_name}/output/timelapse.csv"
    full_output_dir = f'/home/yuhaoliu/Data/ISLAND/cities/{city_name}/output_eval_full'

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
        interp = LST_Interp_No_NLCD(root=root_, target_date=d)  # in ablation mode
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
        interp.run_interpolation()  # run ablation method
        loss, error_map = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
        interp.save_error_frame(mask=added_occlusion, suffix='st')
        print(f'MAE loss over synthetic occluded areas = {loss:.3f}')
        log += [(d, loss, np.count_nonzero(added_occlusion))]
    df = pd.DataFrame(log, columns=['target_date', 'mae', 'synthetic occlusion percentage'])
    df.to_csv(log_fpath, index=False)
    print('csv file saved to ', log_fpath)


def run_fill_average(city_name, dates, resume=False):
    # root_ = f'./data/{city_name}'
    # log_fpath = f"./data/{city_name}/output/timelapse.csv"
    # full_output_dir = f'./data/{city_name}/output_eval_full'

    root_ = '/home/yuhaoliu/Data/ISLAND/cities/{}'.format(city_name)
    log_fpath = f"/home/yuhaoliu/Data/ISLAND/cities/{city_name}/output/timelapse.csv"
    full_output_dir = f'/home/yuhaoliu/Data/ISLAND/cities/{city_name}/output_eval_full'

    assert p.exists(full_output_dir)
    log = []
    for d in dates:
        if resume:
            existing_output_files = os.listdir(p.join(root_, 'output'))
            current_date_files = [f for f in existing_output_files if d in f]
            if len(current_date_files) > 0:
                print(f'Found outputs for date {d}. Skipped.')
                continue
        yprint(f'Evaluating {d} with naive average')
        interp = LST_Interp_Fill_Average(root=root_, target_date=d)
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
        interp.fill_average()  # run ablation method
        loss, error_map = interp.calc_loss_hybrid(metric='mae', synthetic_only_mask=added_occlusion)
        interp.save_error_frame(mask=added_occlusion, suffix='st')
        print(f'MAE loss over synthetic occluded areas = {loss:.3f}')
        log += [(d, loss, np.count_nonzero(added_occlusion))]
    df = pd.DataFrame(log, columns=['target_date', 'mae', 'synthetic occlusion percentage'])
    df.to_csv(log_fpath, index=False)
    print('csv file saved to ', log_fpath)


def parse_csv_dates(region_dir: str):
    assert p.exists(region_dir)
    df = pd.read_csv(p.join(region_dir, 'metadata.csv'))
    assert df is not None, 'csv file not found.'
    dates = df['date'].values.tolist()
    dates = [str(d) for d in dates]
    return dates


def ablation(city_name, occlusion_size=250, num_occlusions=10):
    region_dir = f'/home/yuhaoliu/Data/ISLAND/cities/{city_name}/'
    out_dir = f'/home/yuhaoliu/Data/ISLAND/cities/{city_name}/output/'
    yprint(f'Performing ablation study on {city_name} with up to '
           f'{num_occlusions} random occlusions of size {occlusion_size}')
    dates = parse_csv_dates(region_dir)

    if p.exists(out_dir):
        raise FileExistsError('Output directory already exists. Please move the files elsewhere by '
                              'renaming the folder.')
    run_island(city_name, occlusion_size=occlusion_size, num_occlusions=num_occlusions, resume=False)
    move_output_to(region_dir=region_dir, to_dir='output_eval_full')
    run_no_nlcd(city_name, dates, resume=False)
    move_output_to(region_dir=region_dir, to_dir='output_eval_no_nlcd')
    run_fill_average(city_name, dates, resume=False)
    move_output_to(region_dir=region_dir, to_dir='output_naive_average')
    alert(f'Ablation study for region {city_name} finished processing.')
    print('--------------------------------------------')
    yprint(f'Ablation study for {city_name} with size = {occlusion_size} num = {num_occlusions} is complete.')


@capture_stdout
def calc_error_ablation(city_name, output_dir, mode):
    calc_error_from_outputs(city_name=city_name, output_dir=output_dir, mode=mode)


def run_ablation(city_name, occlusion_size, num_occlusions):
    output_path = p.join('/home/yuhaoliu/Data/ISLAND/cities/', city_name)
    # ablation(city_name, occlusion_size=occlusion_size, num_occlusions=num_occlusions)
    # calc_error_ablation(city_name, p.join(output_path, 'output_eval_full'), 'full')
    # calc_error_ablation(city_name, p.join(output_path, 'output_eval_full'), 'spatial')
    # calc_error_ablation(city_name, p.join(output_path, 'output_eval_full'), 'temporal')
    # calc_error_ablation(city_name, p.join(output_path, 'output_eval_no_nlcd'), 'full')
    calc_error_ablation(city_name, p.join(output_path, 'output_naive_average'), None)


if __name__ == '__main__':
    # run_ablation('Houston', occlusion_size=250, num_occlusions=10) # finished, recorded
    run_ablation('Houston', occlusion_size=750, num_occlusions=3) # op0
    # run_ablation('Phoenix', occlusion_size=500, num_occlusions=1) # finished, recorded
    # run_ablation('Denver', occlusion_size=50, num_occlusions=1) # finished, recorded
    # run_ablation('New York', occlusion_size=100, num_occlusions=2) # finished, recorded
    # run_ablation('Jacksonville', occlusion_size=75, num_occlusions=2) # finished, recorded

