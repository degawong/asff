'''
Author: your name
Date: 1970-01-01 00:00:00
LastEditTime: 2021-01-15 02:32:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ASFFNet/train-asffnet.py
'''
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import torchvision

import itertools
import numpy as np
import skimage.io as io

from network import Loss
from network import ASFFNet
from dataset import ASFFDataSet

device = 'cuda'

def to_torch_script(network):
    # export torch script
    _1_input = torch.randn(
        [1, 3, 256, 256]
    )
    _2_input = torch.randn(
        [1, 3, 256, 256]
    )
    _3_input = torch.randn(
        [1, 1, 256, 256]
    )
    traced_script_module = torch.jit.trace(network, (_1_input.to(device), _2_input.to(device), _3_input.to(device)))
    traced_script_module.save('./checkpoint/torch_script/final.pt')    

def train_process():
    parser = argparse.ArgumentParser(description='ASFFNet by Pytorch')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--learning_rate', '-lr', type=int, default=1e-5,
                        help='learning rate for Adam')
    parser.add_argument('--snapshot_interval', type=int, default=5000,
                        help='Interval of snapshot to generate image')
    parser.add_argument('--degraded_directory', type=str, default='train/degraded_image',
                        help='degraded image directory for train')
    parser.add_argument('--guidance_directory', type=str, default='train/guidance_image',
                        help='guidance image directory for train')
    parser.add_argument('--mask_driectory', type=str, default='train/mask_image',
                        help='mask image directory for train')
    parser.add_argument('--albedo_directory', type=str, default='train/albedo_image',
                        help='albedo image directory for train')
    parser.add_argument('--groundtruth_directory', type=str, default='train/groundtruth_image',
                        help='ground truth image directory for train')
    parser.add_argument('--resume', default=None,
                        help='model state path to load for reuse')
    args = parser.parse_args()

    data = ASFFDataSet(
        args.degraded_directory, 
        args.guidance_directory, 
        args.mask_driectory,
        args.albedo_directory, 
        args.groundtruth_directory
    )
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    asff_network = ASFFNet().to(device)
    asff_network.train()
    asff_loss = Loss().to(device)
    asff_loss.train()
    if args.resume is not None:
	    asff_network.load_state_dict(torch.load(args.resume))
    optimizer = optim.Adam(
        itertools.chain(
            asff_network.parameters(),
            asff_loss.parameters()
        ),
        lr=args.learning_rate, betas=(0.5, 0.999)
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.88)
    for e in range(0, args.epoch):
        for i, (degraded_image, guidance_image, mask_image, albedo_image, groundtruth_image) in enumerate(loader, 0):
            image = asff_network(
                degraded_image.to(device),
                guidance_image.to(device),
                mask_image.to(device),
            )
            loss = asff_loss(
                image,
                mask_image.to(device),
                albedo_image.to(device),
                groundtruth_image.to(device),
            )
            if i % args.snapshot_interval == 0:
                torchvision.utils.save_image(image, "./checkpoint/image/{:03d}_{:05d}.png".format(e, i))    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch : {}; iteration {:08d}; loss : {:f}".format(e, i, loss.item()))
        torch.save(asff_network.state_dict(), './checkpoint/model/{:03d}_epoch.pth'.format(e))
        scheduler.step()

if __name__ == "__main__":
    train_process()