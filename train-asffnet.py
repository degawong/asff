'''
Author: your name
Date: 1970-01-01 00:00:00
LastEditTime: 2021-02-20 15:08:33
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
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import itertools
import numpy as np
import skimage.io as io

import loss as gan
from network import asff_network
from dataset import asff_dataset

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
    parser = argparse.ArgumentParser(description='asff network in Pytorch')
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--learning_rate', '-lr', type=int, default=1e-5,
                        help='learning rate for Adam')
    parser.add_argument('--snapshot_interval', type=int, default=50,
                        help='Interval of snapshot to generate image')
    parser.add_argument('--degraded_directory', type=str, default='/home/datas/storageserver/wgq2719-data/ASFF/train/degraded_image',
                        help='degraded image directory for train')
    parser.add_argument('--guidance_directory', type=str, default='/home/datas/storageserver/wgq2719-data/ASFF/train/guidance_image',
                        help='guidance image directory for train')
    parser.add_argument('--mask_driectory', type=str, default='/home/datas/storageserver/wgq2719-data/ASFF/train/mask_image',
                        help='mask image directory for train')
    parser.add_argument('--albedo_directory', type=str, default='/home/datas/storageserver/wgq2719-data/ASFF/train/albedo_image',
                        help='albedo image directory for train')
    parser.add_argument('--groundtruth_directory', type=str, default='/home/datas/storageserver/wgq2719-data/ASFF/train/groundtruth_image',
                        help='ground truth image directory for train')
    parser.add_argument('--resume', default=None,
                        help='model state path to load for reuse')
    args = parser.parse_args()

    data = asff_dataset(
        args.degraded_directory, 
        args.guidance_directory, 
        args.mask_driectory,
        args.albedo_directory, 
        args.groundtruth_directory
    )
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    
    network = asff_network().to(device)
    network.train()
    
    degraded_operation = gan.degraded_operation().to(device)
    degraded_operation.train()

    discriminator = gan.discriminator().to(device)
    discriminator.train()

    generator_loss = gan.generator_loss().to(device)
    discriminator_loss = gan.discriminator_loss().to(device)
    
    if args.resume is not None:
	    network.load_state_dict(torch.load(args.resume))

    generator_optimizer = optim.Adam(
        itertools.chain(
            network.parameters(),
            generator_loss.parameters(),
            degraded_operation.parameters()
        ),
        lr=args.learning_rate, betas=(0.5, 0.999)
    )

    discriminator_optimizer = optim.Adam(
        itertools.chain(
            discriminator.parameters()
        ),
        lr=args.learning_rate, betas=(0.5, 0.999)
    )

    writer = SummaryWriter('checkpoint/log')

    for e in range(0, args.epoch):
        for i, (degraded_image, guidance_image, mask_image, albedo_image, groundtruth_image) in enumerate(loader, 0):
            
            generated_image = network(
                degraded_image.to(device),
                guidance_image.to(device),
                mask_image.to(device),
            )

            fake_tag = torch.autograd.Variable(torch.Tensor(degraded_image.size(0), 1).fill_(0.0), requires_grad=False)
            real_tag = torch.autograd.Variable(torch.Tensor(degraded_image.size(0), 1).fill_(1.0), requires_grad=False)

            encodered_image = degraded_operation(generated_image)

            generator_optimizer.zero_grad()
            generated_tag = discriminator(generated_image)
            g_loss = generator_loss(
                degraded_image.to(device),
                generated_image,
                guidance_image.to(device),
                encodered_image,
                generated_tag,
                mask_image.to(device)
            )
            g_loss.backward(retain_graph = True)
            generator_optimizer.step()

            discriminator_optimizer.zero_grad()
            fake = discriminator(generated_image)
            real = discriminator(guidance_image.to(device))
            d_loss = discriminator_loss(
                generated_image,
                guidance_image.to(device),
                [fake, real],
                mask_image.to(device)
            )
            d_loss.backward()
            discriminator_optimizer.step()

            writer.add_scalar('generator_loss', g_loss, global_step = e * len(degraded_image) + i)
            writer.add_scalar('discriminator_loss', d_loss, global_step = e * len(degraded_image) + i)
            writer.add_image(
                "generated_image",
                torchvision.utils.make_grid(generated_image),
                global_step = e * len(degraded_image) + i,
                walltime = None,
                dataformats='CHW'
            )
            writer.add_graph(
                network,
                input_to_model = (
                    degraded_image.to(device),
                    guidance_image.to(device),
                    mask_image.to(device), 
                ),
                verbose = False,
            )

            if i % args.snapshot_interval == 0:
                torchvision.utils.save_image(generated_image, "./checkpoint/image/{:03d}_{:05d}.png".format(e, i))    
            print("epoch : {}; iteration {:08d}; g_loss : {:f}; d_loss : {:f}".format(e, i, g_loss.item(), d_loss.item()))
        torch.save(network.state_dict(), './checkpoint/model/{:03d}_epoch.pth'.format(e))

if __name__ == "__main__":
    train_process()