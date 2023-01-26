"""
Conduct ablation study with synthetic occlusion for a specified city
"""
__author__ = 'yuhao liu'

import os
import os.path as p
import numpy as np
import pandas as pd


def ablation(city_name):
    root_ = f'./data/{city_name}/'
    df = pd.read_csv(p.join(root_, 'metadata.csv'))
    dates = df['date'].values.tolist()
    dates = [str(d) for d in dates]


def main():
    pass


if __name__ == '__main__':
    main()
