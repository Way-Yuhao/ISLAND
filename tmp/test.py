# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt
# import pandas as pd
# from util.helper import deprecated, time_func
# import ee
# import geemap
# import wandb
import argparse

# @time_func
def main():
    parser = argparse.ArgumentParser(description='Process specify city name.')
    parser.add_argument('-c', nargs='+', required=True,
                        help='Process specify city name.')
    args = parser.parse_args()
    print(args.c)
    print(args.c[0])
    CITY_NAME = ""
    for entry in args.c:
        CITY_NAME += entry + " "
    CITY_NAME = CITY_NAME[:-1]
    # CITY_NAME = str(CITY_NAME)[:-1]
    print(CITY_NAME)

if __name__ == '__main__':
    main()
