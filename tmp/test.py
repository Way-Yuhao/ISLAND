# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt
# import pandas as pd
# from util.helper import deprecated, time_func
from matplotlib import pyplot as plt
# import ee
# import geemap
# import wandb
import cv2
import argparse


def main():
    emis = cv2.imread('../data/Houston/emis/LC08_ST_EMIS_20220623.tif', -1) * 0.0001
    plt.imshow(emis)
    plt.show()


if __name__ == '__main__':
    main()
