from config import *
import torch
import torch.nn as nn
import argparse
import re
from models.model import DeepLabV3PlusRGB
import numpy as np


def main():
    str = 'LC08_cirrus_houston_20190311.tif'
    print(str[-12:-4])


if __name__ == '__main__':
    main()
