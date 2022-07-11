from config import *
import torch
import torch.nn as nn
import argparse
import re
from models.model import DeepLabV3PlusRGB
import numpy as np
import mss


def main():
    path = '../data/export/output/r_20181221_20191106.png.npy'
    a = np.load(path)
    print(a)


if __name__ == '__main__':
    main()
