import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from util.helper import deprecated
import ee
import geemap

def main():
    ee.Initialize()
    img = ee.Image('LANDSAT/LC08/C02/T1_TOA/LC08_025039_20200210').select('B0io')
    print(img)


if __name__ == '__main__':
    main()
