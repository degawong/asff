'''
Author: your name
Date: 1970-01-01 00:00:00
LastEditTime: 2021-02-20 15:08:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ASFFNet/train-asffnet.py
'''

"""
TODO: use replay buffer improve train effect
"""

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
    parser.add_argument('--batch_size', '-b', type=int, default=1,
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
    )
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    
    network = asff_network().to(device)
    network.train()
    
    degraded_operation = gan.degraded_operation().to(device)
    degraded_operation.train()

    discriminator_x = gan.discriminator().to(device)
    discriminator_x.train()

    discriminator_y = gan.discriminator().to(device)
    discriminator_y.train()
    
    if args.resume is not None:
	    network.load_state_dict(torch.load(args.resume))

    generator_optimizer = optim.Adam(
        itertools.chain(
            network.parameters(),
            degraded_operation.parameters()
        ),
        lr=args.learning_rate, betas=(0.5, 0.999)
    )

    discriminator_x_optimizer = optim.Adam(
        itertools.chain(
            discriminator_x.parameters()
        ),
        lr=args.learning_rate, betas=(0.5, 0.999)
    )

    discriminator_y_optimizer = optim.Adam(
        itertools.chain(
            discriminator_y.parameters()
        ),
        lr=args.learning_rate, betas=(0.5, 0.999)
    )

    l1_gan = torch.nn.MSELoss()
    l1_cycle = torch.nn.L1Loss()
    l1_identity = torch.nn.L1Loss()

    writer = SummaryWriter('checkpoint/log')
 
    for e in range(0, args.epoch):
        for i, (_up_degraded, _up_guidance, _up_mask, _down_degraded, _down_guidance, _down_mask) in enumerate(loader, 0):
            # pixel gan
            up_degraded = _up_degraded.to(device)
            up_guidance = _up_guidance.to(device)
            up_mask = _up_mask.to(device)
            down_degraded = _down_degraded.to(device)
            down_guidance = _down_guidance.to(device)
            down_mask = _down_mask.to(device)

            # real = torch.ones_like(up_degraded, requires_grad = False)
            # fake = torch.zeros_like(up_degraded, requires_grad = False)
            fake = torch.autograd.Variable(torch.Tensor(down_mask.size()).fill_(1.0), requires_grad = False).to(device)
            real = torch.autograd.Variable(torch.Tensor(down_mask.size()).fill_(1.0), requires_grad = False).to(device)

            up_generated = network(
                up_degraded,
                up_guidance,
                up_mask,
            )
            up_recovered = degraded_operation(up_generated)

            down_generated = degraded_operation(down_degraded)
            down_recovered = network(
                down_generated,
                down_guidance,
                down_mask,
            )

            # generator
            generator_optimizer.zero_grad()
            ## identity loss
            x_clone = degraded_operation(up_degraded)
            # need or not ?
            y_clone = network(
                down_guidance,
                down_guidance,
                down_mask,
            )
            # identity_loss_x = l1_identity(up_degraded, x_clone)
            # identity_loss_y = l1_identity(down_guidance, y_clone)
            # identity_loss = identity_loss_x + identity_loss_y
            identity_loss = l1_identity(up_degraded, x_clone) + l1_identity(down_guidance, y_clone)
            ## gan loss
            # real_y = discriminator_y(up_generated)
            # gan_loss_y = l1_gan(real_y, real)
            # real_x = discriminator_x(down_generated)
            # gan_loss_x = l1_gan(real_x, real)
            # adversarial_loss = gan_loss_x + gan_loss_y
            adversarial_loss = l1_gan(discriminator_x(down_generated), real) + l1_gan(discriminator_y(up_generated), real)
            ## cycle loss
            # cycle_loss_x = l1_cycle(up_degraded, up_recovered)
            # cycle_loss_y = l1_cycle(down_guidance, down_recovered)
            # cycle_loss = cycle_loss_x + cycle_loss_y
            cycle_loss = l1_cycle(up_degraded, up_recovered) + l1_cycle(down_guidance, down_recovered)

            generator_loss = identity_loss + adversarial_loss + cycle_loss
            generator_loss.backward()
            
            generator_optimizer.step()

            # discriminator_x
            discriminator_x_optimizer.zero_grad()
            discriminator_x_loss = l1_gan(discriminator_x(up_degraded), real) + l1_gan(discriminator_x(down_generated.detach()), fake)
            discriminator_x_loss.backward()
            discriminator_x_optimizer.step()

            # discriminator_y
            discriminator_y_optimizer.zero_grad()
            discriminator_y_loss = l1_gan(discriminator_y(down_guidance), real) + l1_gan(discriminator_y(up_generated.detach()), fake)
            discriminator_y_loss.backward()
            discriminator_y_optimizer.step()

            writer.add_scalar(
                'generator_loss',
                generator_loss,
                global_step = e * len(up_degraded) + i
            )
            writer.add_scalar(
                'discriminator_x_loss',
                discriminator_x_loss,
                global_step = e * len(up_degraded) + i
            )
            writer.add_scalar(
                'discriminator_y_loss',
                discriminator_y_loss,
                global_step = e * len(up_degraded) + i
            )
            writer.add_scalar(
                'loss',
                generator_loss + discriminator_x_loss + discriminator_y_loss,
                global_step = e * len(up_degraded) + i
            )
            writer.add_image(
                "up_generated",
                torchvision.utils.make_grid(torch.cat((up_degraded, up_generated, up_recovered))),
                global_step = e * len(up_degraded) + i,
                walltime = None,
                dataformats='CHW',
            )
            writer.add_image(
                "down_generated",
                torchvision.utils.make_grid(torch.cat((down_guidance, down_generated, down_recovered))),
                global_step = e * len(up_degraded) + i,
                walltime = None,
                dataformats='CHW',
            )
            # writer.add_graph(
                # network,
                # input_to_model = (
                    # up_degraded,
                    # up_guidance,
                    # up_mask, 
                # ),
                # verbose = False,
            # )
            # writer.add_graph(
                # degraded_operation,
                # input_to_model = (
                    # up_generated,
                # ),
                # verbose = False,
            # )
            if i % args.snapshot_interval == 0:
                torchvision.utils.save_image(torch.cat((up_degraded, up_generated, up_recovered)), "./checkpoint/image/{:03d}_{:05d}_up.png".format(e, i))
                torchvision.utils.save_image(torch.cat((down_guidance, down_generated, down_recovered)), "./checkpoint/image/{:03d}_{:05d}_down.png".format(e, i))
            print("epoch : {}; iteration {:08d}; g_loss : {:f}; x_loss : {:f}; y_loss : {:f}".format(e, i, generator_loss.item(), discriminator_x_loss.item(), discriminator_y_loss.item()))
        torch.save(network.state_dict(), './checkpoint/model/{:03d}_epoch.pth'.format(e))

if __name__ == "__main__":
    train_process()