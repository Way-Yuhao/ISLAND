import argparse
from batch_eval import solve_all_bt, move_bt, compute_st_for_all


def process_city():
    """
    Computes brightness temperature and surface temperature for a given city. Require inputs to be downloaded
    in advance.
    :return:
    """
    parser = argparse.ArgumentParser(description='Process specify city name.')
    parser.add_argument('-c', nargs='+', required=True,
                        help='Process specify city name.')
    args = parser.parse_args()
    CITY_NAME = ""
    for entry in args.c:
        CITY_NAME += entry + " "
    CITY_NAME = CITY_NAME[:-1]
    solve_all_bt(city_name=CITY_NAME)  # solve for brightness temperature
    move_bt(city_name=CITY_NAME)
    compute_st_for_all(city_name=CITY_NAME)  # solve for surface temperature


def main():
    process_city()


if __name__ == '__main__':
    main()
