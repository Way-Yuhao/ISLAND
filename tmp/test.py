import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from util.helper import deprecated, time_func
import ee
import geemap
import wandb


@time_func
def main():
    wandb.init()
    wandb.alert(
        title='Download finished',
        text='yo'
    )


if __name__ == '__main__':
    main()
