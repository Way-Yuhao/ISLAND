__author__ = "Yuhao Liu"

import argparse
import numpy as np
import os.path as p
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
import torchvision
from torch.utils.tensorboard import SummaryWriter
import geemap
from tqdm import tqdm
from earth_engine_loader import LocalCropLoader
from config import *
from models.model import RGBSegNet
from models.model import DeepLabV3PlusRGB
from util.focal_loss import FocalLoss


"""Global Parameters"""
input_meta = LANDSAT8_META
label_meta = NLCD_2019_META
bounding_box = HOUSTON_BOUNDING_BOX
ee_buffer_dir = "/mnt/data1/yl241/datasets/ee_buffer/houston_train_dev_random_cloud"
network_weight_path = "./weight/"
CUDA_DEVICE = None  # parsed in main
version = None  # defined in main
model_name = None  # defined in main
num_workers_train = 16
batch_size = 16
validation_split = VALIDATION_SPLIT

"Hyper Parameters"
init_lr = 1e-4
epoch = 1000
cloud_prob = LANDSAT8_META['avg_cloud_cover']
freq = np.array([cloud_prob,  # 0
                 (1 - cloud_prob) * (.28 + .16 + .09 + .05),  # 1
                 (1 - cloud_prob) * .007,  # 2
                 (1 - cloud_prob) * (.005 + .004 + .004),  # 3
                 (1 - cloud_prob) * (.10 + .02 + .02 + .01),  # 4
                 (1 - cloud_prob) * (.09 + .04)])  # 5
alphas = 1 - freq  # for focal loss

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


def load_data(mode: str):
    assert mode in ['train', 'dev']
    data_loader = torch.utils.data.DataLoader(
        LocalCropLoader(root=p.join(ee_buffer_dir, mode)),
        batch_size=batch_size, num_workers=num_workers_train)
    return data_loader


def load_network_weights(net, path):
    """
    loads pre-trained weights from path and prints message
    :param net: pytorch network object
    :param path: path to load network weights
    :return: None
    """
    print("loading pre-trained weights from {}".format(path))
    net.load_state_dict(torch.load(path))
    return



def save_network_weights(net, ep=None):
    filename = network_weight_path + "{}{}_epoch_{}.pth".format(model_name, version, ep)
    torch.save(net.state_dict(), filename)
    print("network weights saved to ", filename)
    return


def compute_loss(output, label):
    """
    computes the cross entropy loss
    :param output:
    :param label:
    :return:
    """
    # consider adding class balanced weights
    # output = output['out']
    # output_predictions = output.argmax(dim=1).unsqueeze(1)
    ce_criterion = nn.CrossEntropyLoss()
    ce_loss = ce_criterion(output, label)
    return ce_loss

def compute_focal_loss(output, label):
    fl_criterion = FocalLoss(alpha=torch.from_numpy(alphas).to(CUDA_DEVICE, dtype=torch.float),
                             gamma=2, reduction='mean')
    fc_loss = fl_criterion(output, label)
    return fc_loss

def extract_rgb_bands(hs_input):
    r, g, b = input_meta['rgb_bands']
    r, g, b = input_meta['selected_bands'].index(r), input_meta['selected_bands'].index(g), \
              input_meta['selected_bands'].index(b)
    rgb_img = torch.stack((hs_input[:, r, :, :], hs_input[:, g, :, :], hs_input[:, b, :, :]), dim=1)
    # rgb_img[rgb_img > .3] = .3
    # rgb_img[rgb_img < 0] = 0
    return rgb_img


def extract_mask(hs_input):
    m = input_meta['selected_bands'].index('ST_CDIST')
    cloud_mask = hs_input[:, m, :, :].unsqueeze(1)
    return cloud_mask


def label_vis(label):
    label_vis_r = torch.zeros_like(label, dtype=torch.uint8)
    label_vis_g = torch.zeros_like(label, dtype=torch.uint8)
    label_vis_b = torch.zeros_like(label, dtype=torch.uint8)
    for i in range(VIS_PARAM['label_min'], VIS_PARAM['label_max'] + 1):
        label_vis_r[label == i] = VIS_PARAM['label_palette'][i - 1][0]
        label_vis_g[label == i] = VIS_PARAM['label_palette'][i - 1][1]
        label_vis_b[label == i] = VIS_PARAM['label_palette'][i - 1][2]

    # label_vis = torch.hstack((label_vis_r, label_vis_g, label_vis_b))
    label_vis = torch.stack((label_vis_r, label_vis_g, label_vis_b), dim=1)
    return label_vis


def output_vis(output):
    # output = (torch.round(output)).type(torch.uint8)  # round to nearest int
    output_vis_r = torch.zeros_like(output, dtype=torch.uint8)
    output_vis_g = torch.zeros_like(output, dtype=torch.uint8)
    output_vis_b = torch.zeros_like(output, dtype=torch.uint8)
    for i in range(VIS_PARAM['label_min'], VIS_PARAM['label_max'] + 1):
        output_vis_r[output == i] = VIS_PARAM['label_palette'][i - 1][0]
        output_vis_g[output == i] = VIS_PARAM['label_palette'][i - 1][1]
        output_vis_b[output == i] = VIS_PARAM['label_palette'][i - 1][2]
    output_vis = torch.stack((output_vis_r, output_vis_g, output_vis_b), dim=1)
    return output_vis


def tensorboard_vis(tb, step, mode='train', input_=None, output=None, label=None, display_mask=False):
    output = output.argmax(dim=1).type(torch.uint8)  # output predictions
    tb.add_histogram(f"{mode}/output_", output, global_step=step)
    tb.add_histogram(f"{mode}/label_", label, global_step=step)
    if input_ is not None:
        input_img_grid = torchvision.utils.make_grid(extract_rgb_bands(input_), normalize=True)
        tb.add_image(f"{mode}/input", input_img_grid, global_step=step)
    if display_mask and input_ is not None:
        input_mask_grid = torchvision.utils.make_grid(extract_mask(input_), normalize=True)
        tb.add_image(f"{mode}/cloud_mask", input_mask_grid, global_step=step)
    if output is not None:
        if mode == 'train':  # no threshold in visualization
            output_img_grid = torchvision.utils.make_grid(output_vis(output))
            tb.add_image(f"{mode}/output", output_img_grid, global_step=step)
        elif mode == 'dev':  # apply threshold in visualization
            output_img_grid = torchvision.utils.make_grid(output_vis(output))
            tb.add_image(f"{mode}/output", output_img_grid, global_step=step)
    if label is not None:
        label_img_grid = torchvision.utils.make_grid(label_vis(label))
        tb.add_image(f"{mode}/label", label_img_grid, global_step=step)
    return


def train(net, tb):
    global CUDA_DEVICE
    print_params()
    net.train()

    train_loader = load_data(mode='train')
    dev_loader = load_data(mode='dev')
    print(f"size of train set = {len(train_loader)} mini-batches | size of dev set = {len(dev_loader)} mini-batches")

    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.)  # do nothing
    train_num_mini_batches, dev_num_mini_batches = len(train_loader), len(dev_loader)
    running_train_loss, running_dev_loss = 0., 0.
    train_input, train_output, train_label = None, None, None
    for ep in range(epoch):
        print("{}/{} | Step {}".format(model_name, version, ep))
        train_iter, dev_iter = iter(train_loader), iter(dev_loader)
        # TRAIN
        for _ in tqdm(range(train_num_mini_batches)):
            train_input, train_label = train_iter.next()
            train_input, train_label = train_input.to(CUDA_DEVICE, dtype=torch.float), \
                                       train_label.to(CUDA_DEVICE, dtype=torch.long)
            train_output = net(train_input)
            train_loss = compute_focal_loss(train_output, train_label)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
        # DEV
        with torch.no_grad():
            for _ in range(dev_num_mini_batches):
                dev_input, dev_label = dev_iter.next()
                dev_input, dev_label = dev_input.to(CUDA_DEVICE, dtype=torch.float), \
                                       dev_label.to(CUDA_DEVICE, dtype=torch.long)
                dev_output = net(dev_input)
                dev_loss = compute_focal_loss(dev_output, dev_label)
                running_dev_loss += dev_loss.item()
        cur_train_loss = running_train_loss / train_num_mini_batches
        cur_dev_loss = running_dev_loss / dev_num_mini_batches
        print("train loss = {:.4} | val loss = {:.4}".format(cur_train_loss, cur_dev_loss))
        tb.add_scalar('loss/train', cur_train_loss, ep)
        tb.add_scalar('loss/dev', cur_dev_loss, ep)
        tb.add_scalar('loss/lr', scheduler._last_lr[0], ep)
        if ep % 10 == 0:
            tensorboard_vis(tb, mode='dev', step=ep, input_=dev_input, output=dev_output, label=dev_label,
                            display_mask=True)
        if ep % 50 == 0:
            save_network_weights(net, ep)
        running_train_loss, running_dev_loss = 0., 0.
        scheduler.step()

    print("finished training")
    save_network_weights(net, ep=f"{epoch}_FINAL")


def parse_args():
    parser = argparse.ArgumentParser(description='Specify target GPU, else the one defined in config.py will be used.')
    parser.add_argument('--gpu', type=str, help='cuda:$')
    args = parser.parse_args()
    if args.gpu is not None:
        CUDA_DEVICE = "cuda:{}".format(args.gpu)
    else:
        CUDA_DEVICE = "cpu".format(args.gpu)
    return CUDA_DEVICE

def init_data_parallel(net, CUDA_DEVICE):
    if len(CUDA_DEVICE) <= 6: # 1 or no gpu
        net.to(CUDA_DEVICE)
        return CUDA_DEVICE
    else:
        primary_device = CUDA_DEVICE[:6]  # first gpu
        net = nn.DataParallel(net, device_ids=[int(x) for x in CUDA_DEVICE if x.isdigit()])
        net.to(primary_device)
        return primary_device

def main():
    global version, model_name, CUDA_DEVICE
    model_name, version = "mHyper-SegV3+", "v1.1.3-cont"
    CUDA_DEVICE = parse_args()
    tb = SummaryWriter('./runs/' + model_name + '/' + version)
    net = DeepLabV3PlusRGB(num_classes=NUM_CLASSES, backbone='resnet')
    CUDA_DEVICE = init_data_parallel(net, CUDA_DEVICE)
    param_to_load = p.join(network_weight_path, f'{model_name}v1.1.3_epoch_500_FINAL.pth')
    load_network_weights(net, param_to_load)
    train(net, tb)
    tb.close()


if __name__ == '__main__':
    main()
