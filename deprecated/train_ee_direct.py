__author__ = "Yuhao Liu"

import os
import os.path as p
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import geemap
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from earth_engine_loader import EarthEngineLoader
from config import *
from model import RGBSegNet

"""Global Parameters"""
input_meta = LANDSAT8_META
label_meta = NLCD_2019_META
bounding_box = HOUSTON_BOUNDING_BOX
network_weight_path = "./weight/"
CUDA_DEVICE = None  # parsed in main
version = None  # defined in main
model_name = None  # defined in main
num_workers_train = 1
batch_size = 2

"Hyper Parameters"
init_lr = 1e-4
epoch = 200
max_steps = 500  # maximum number of mini batches used for training

"Visualization Parameters"
rgb_vis_min = 0.0
rgb_vis_max = 0.3


def print_params():
    print("######## Basics ##################")
    print("version: {}".format(version))
    print("Training on {}".format(CUDA_DEVICE))
    print("batch size = ", batch_size)
    print("number of workers = ", num_workers_train)
    print("#################################")


def load_data(map_):
    data_loader = torch.utils.data.DataLoader(
        EarthEngineLoader(root="./", geemap_obj=map_, bounding_box=bounding_box, image_meta=input_meta,
                          label_meta=label_meta),
        batch_size=batch_size, num_workers=num_workers_train)
    return data_loader


def save_network_weights(net, ep=None):
    filename = network_weight_path + "{}{}_epoch_{}.pth".format(model_name, version, ep)
    torch.save(net.state_dict(), filename)
    print("network weights saved to ", filename)
    return


def compute_loss(output, label):
    output = output['out']
    # output_predictions = output.argmax(dim=1).unsqueeze(1)
    ce_criterion = nn.CrossEntropyLoss()
    ce_loss = ce_criterion(output, label)
    return ce_loss


def extract_rgb_bands(hs_input):
    r, g, b = input_meta['rgb_bands']
    r, g, b = input_meta['selected_bands'].index(r), input_meta['selected_bands'].index(g), \
              input_meta['selected_bands'].index(b)
    rgb_img = torch.stack((hs_input[:, r, :, :], hs_input[:, g, :, :], hs_input[:, b, :, :]), dim=1)
    # rgb_img[rgb_img > .3] = .3
    # rgb_img[rgb_img < 0] = 0
    return rgb_img


def label_vis(label):
    # label = (label.type(torch.float32) - VIS_PARAM['label_min']) / (VIS_PARAM['label_max'] - VIS_PARAM['label_min'])
    # return label

    # label_stacked = torch.hstack((label, label, label))
    label_vis_r = torch.zeros_like(label, dtype=torch.uint8)
    label_vis_g = torch.zeros_like(label, dtype=torch.uint8)
    label_vis_b = torch.zeros_like(label, dtype=torch.uint8)
    for i in range(VIS_PARAM['label_min'], VIS_PARAM['label_max'] + 1):
        label_vis_r[label == i] = VIS_PARAM['label_palette'][i - 1][0]
        label_vis_g[label == i] = VIS_PARAM['label_palette'][i - 1][1]
        label_vis_b[label == i] = VIS_PARAM['label_palette'][i - 1][2]

    label_vis = torch.hstack((label_vis_r, label_vis_g, label_vis_b))
    return label_vis


def tensorboard_vis(tb, step, mode='train', input_=None, output=None, label=None):
    # tb.add_histogram(f"{mode}/output_", output, global_step=step)
    tb.add_histogram(f"{mode}/label_", label, global_step=step)
    if input_ is not None:
        input_img_grid = torchvision.utils.make_grid(extract_rgb_bands(input_), normalize=True)
        tb.add_image(f"{mode}/input", input_img_grid, global_step=step)
    if output is not None:
        if mode == 'train':  # no threshold in visualization
            output_img_grid = torchvision.utils.make_grid(output)
            tb.add_image(f"{mode}/output", output_img_grid, global_step=step)
        elif mode == 'dev':  # apply threshold in visualization
            raise NotImplementedError()
    if label is not None:
        label_img_grid = torchvision.utils.make_grid(label_vis(label))
        tb.add_image(f"{mode}/label", label_img_grid, global_step=step)
    return


def train(net, tb, map_):
    print_params()
    net.to(CUDA_DEVICE)
    net.train()
    train_loader = load_data(map_)

    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.)  # do nothing
    train_num_mini_batches = len(train_loader)
    running_train_loss = 0.
    train_input, train_output, train_label = None, None, None
    for ep in range(epoch):
        print("{}-{} | Step {}".format(model_name, version, ep))
        train_iter = iter(train_loader)
        # TRAIN
        for step in tqdm(range(train_num_mini_batches)):
            train_input, train_label = train_iter.next()
            train_input, train_label = train_input.to(CUDA_DEVICE, dtype=torch.float), \
                                       train_label.to(CUDA_DEVICE, dtype=torch.long)
            train_output = net(train_input)
            train_loss = compute_loss(train_output, train_label)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
            tb.add_scalar('loss/train_micro', train_loss.item(), ep * train_num_mini_batches + step)

        cur_train_loss = running_train_loss / train_num_mini_batches
        print("train loss = {:.4}".format(cur_train_loss))
        tb.add_scalar('loss/train', cur_train_loss, ep)
        tb.add_scalar('loss/lr', scheduler._last_lr[0], ep)
        if step % 5 == 0:
            tensorboard_vis(tb, step=step, input_=train_input, label=train_label)
        running_train_loss = 0
        scheduler.step()

    print("finished training")
    # TODO: save network


def parse_args():
    parser = argparse.ArgumentParser(description='Specify target GPU, else the one defined in config.py will be used.')
    parser.add_argument('--gpu', type=int, help='cuda:$')
    args = parser.parse_args()
    if args.gpu is not None:
        CUDA_DEVICE = "cuda:{}".format(args.gpu)
    else:
        CUDA_DEVICE = "cpu".format(args.gpu)
    return CUDA_DEVICE


def main():
    global version, model_name, CUDA_DEVICE
    model_name, version = "RGBSeg", "v0.3.2-test3"
    CUDA_DEVICE = parse_args()
    param_to_load = None
    tb = SummaryWriter('./runs/' + model_name + '-' + version)
    map_ = geemap.Map()
    net = RGBSegNet()
    train(net, tb, map_)
    tb.close()


if __name__ == '__main__':
    main()
