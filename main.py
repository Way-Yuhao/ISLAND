import os
import os.path as p
import time
import wandb
import argparse
import shutil
from batch_eval import solve_all_bt, move_bt, compute_st_for_all
from util.helper import get_season, rprint, yprint, time_func


def process_city():
    """
    Computes brightness temperature and surface temperature for a given city. Require inputs to be downloaded
    in advance.
    :return:
    """
    wandb.init()
    parser = argparse.ArgumentParser(description='Process specify city name.')
    parser.add_argument('-c', nargs='+', required=True,
                        help='Process specify city name.')
    parser.add_argument('-r', required=False, action='store_true',
                        help='Toggle to resume from previous run. Will not overwrite files.')
    args = parser.parse_args()
    RESUME = args.r
    CITY_NAME = ""
    for entry in args.c:
        CITY_NAME += entry + " "
    CITY_NAME = CITY_NAME[:-1]
    if RESUME:
        yprint('WARNING: resume mode is on')
        if p.exists(f'./data/{CITY_NAME}/output_bt'):
            shutil.rmtree(f'./data/{CITY_NAME}/output_bt')
            yprint(f'Removing ./data/{CITY_NAME}/output_bt')
        if p.exists(f'./data/{CITY_NAME}/output_st'):
            shutil.rmtree(f'./data/{CITY_NAME}/output_st')
            yprint(f'Removing ./data/{CITY_NAME}/output_st')
        time.sleep(2)  # allow previous messages to print
    elif p.exists(f'./data/{CITY_NAME}/output'):
        raise FileExistsError(f'Output directory ./data/{CITY_NAME}/output/ already exists'
                              f'please ether turn \'resume\' on or remove the existing '
                              f'directory.')
    solve_all_bt(city_name=CITY_NAME, resume=RESUME)  # solve for brightness temperature
    move_bt(city_name=CITY_NAME)
    compute_st_for_all(city_name=CITY_NAME)  # solve for surface temperature
    wandb.alert(
        title='Interpolation finished',
        text=f'Data for region {CITY_NAME} finished processing.'
    )


def main():
    process_city()


if __name__ == '__main__':
    main()
