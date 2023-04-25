import cv2
from util.helper import save_cmap
import os.path as p
import numpy as np
from matplotlib import pyplot as plt
from util.occlusion_sampler import OcclusionSampler
import random


def test():
    img = cv2.imread('../data/Houston/cloud/LC08_cloud_20170116.tif', -1).astype(np.float32)
    # img = np.flip(img, [0, 1])
    # img = np.rot90(img, 2)
    # plt.imshow(img)
    # plt.show()
    # print(img.shape)
    # img = cv2.resize(img, dsize=(2500, 2000), interpolation=cv2.INTER_LINEAR).astype(bool)
    # # print(img)
    # plt.imshow(img)
    # plt.show()
    print(img.shape)


def test2():
    date_list = [('Houston', '20200414'), ('Austin', '20190816'), ('Oklahoma City', '20180719'),
                 ('San Diego', '20181112')]
    r = [e[0] for e in date_list]
    print(r)

def main():
    data_list = [('Houston', '20200414'), ('Austin', '20190816'), ('Oklahoma City', '20180719'),
                 ('San Diego', '20181112')]
    city_list = ['Houston', 'Austin', 'Oklahoma City', 'San Diego']
    sampler = OcclusionSampler(data_list)
    img = sampler.sample(0.9, (3000, 2000))
    plt.imshow(img)
    plt.show()
    print(img)

if __name__ == '__main__':
    test2()