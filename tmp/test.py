from config import *
import torch
import torch.nn as nn
import argparse
import re
from models.model import DeepLabV3PlusRGB
import numpy as np

def main():
    cloud_prob = LANDSAT8_META['avg_cloud_cover']
    freq = np.array([cloud_prob,
                     (1 - cloud_prob) * (.28 + .16 + .09 + .05),
                     (1 - cloud_prob) * .007,
                     (1 - cloud_prob) * (.005 + .004 + .004),
                     (1 - cloud_prob) * (.10 + .02 + .02 + .01),
                     (1 - cloud_prob) * (.09 + .04)])
    alphas = 1 - freq
    print(alphas)
if __name__ == '__main__':
    main()
