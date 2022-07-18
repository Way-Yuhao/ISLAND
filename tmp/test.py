import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from util.helper import deprecated


def main():
    a = np.array([1, 2, -3, np.NINF, np.nan])
    if np.isnan(a):
        print('hell nah')

if __name__ == '__main__':
    main()
