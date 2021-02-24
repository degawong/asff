'''
Author: your name
Date: 2021-01-04 17:07:53
LastEditTime: 2021-02-19 10:22:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \asff_network\network.py
'''
from typing import ClassVar
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
import numpy as np

class asff_feature(nn.Module):
    """Some Information about asff_feature"""
    def __init__(self):
        super(asff_feature, self).__init__()
        self.__Feature = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', nn.Conv2d(3, 64, 3, 1, 1, bias=True)),
                    ('layer-01', dilated_residual_block(64, 64, 64, 3, 1, 7, 7, 0.2)),
                    ('layer-02', dilated_residual_block(64, 64, 64, 3, 1, 5, 5, 0.2)),
                    ('layer-03', nn.Conv2d(64, 128, 3, 2, 1, bias=True)),
                    ('layer-04', dilated_residual_block(128, 128, 128, 3, 1, 5, 5, 0.2)),
                    ('layer-05', dilated_residual_block(128, 128, 128, 3, 1, 3, 3, 0.2)),
                    ('layer-06', nn.Conv2d(128, 128, 3, 2, 1, bias=True)),
                    ('layer-07', dilated_residual_block(128, 128, 128, 3, 1, 3, 3, 0.2)),
                    ('layer-08', dilated_residual_block(128, 128, 128, 3, 1, 1, 1, 0.2)),
                    ('layer-09', nn.Conv2d(128, 128, 3, 1, 1, bias=True)),
                    ('layer-10', nn.LeakyReLU(0.2))
                ]
            )
        )
    def forward(self, x):
        return self.__Feature(x)

class asff_block(nn.Module):
    """Some Information about asff_block"""
    def __init__(self):
        super(asff_block, self).__init__()
        self.__mask_operation_1 = nn.Conv2d(128, 64, 1, 1, 0, bias=True)
        self.__degraded_operation_1 = nn.Conv2d(128, 64, 1, 1, 0, bias=True)
        self.__guidance_operation_1 = nn.Conv2d(128, 64, 1, 1, 0, bias=True)
        self.__degraded_operation_2 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', nn.Conv2d(128, 128, 3, 1, 1, bias=True)),
                    ('layer-01', nn.BatchNorm2d(128)),
                    ('layer-02', nn.LeakyReLU(0.2)),
                    ('layer-03', nn.Conv2d(128, 128, 1, 1, 0, bias=True)),
                ]
            )
        )
        self.__guidance_operation_2 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', nn.Conv2d(128, 128, 3, 1, 1, bias=True)),
                    ('layer-01', nn.BatchNorm2d(128)),
                    ('layer-02', nn.LeakyReLU(0.2)),
                    ('layer-03', nn.Conv2d(128, 128, 1, 1, 0, bias=True)),
                ]
            )
        )
        self.__mask_operation_2 = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', nn.Conv2d(64 * 3, 128, 3, 1, 1, bias=False)),
                    ('layer-01', nn.BatchNorm2d(128)),
                    ('layer-02', nn.LeakyReLU(0.2)),
                    ('layer-03', nn.Conv2d(128, 128, 3, 1, 1, bias=False)),
                    ('layer-04', nn.BatchNorm2d(128)),
                    ('layer-05', nn.LeakyReLU(0.2)),
                ]
            )
        )

    def forward(self, degraded_feature, guidance_feature, mask_features):
        fd = self.__degraded_operation_1(degraded_feature)
        fg = self.__guidance_operation_1(guidance_feature)
        fm_temp = self.__mask_operation_1(mask_features)

        fm = torch.cat([fd, fg], 1)
        fm = torch.cat([fm, fm_temp], 1)

        fd = self.__degraded_operation_2(degraded_feature)
        fg = self.__guidance_operation_2(guidance_feature)
        fm = self.__mask_operation_2(fm)

        return fd + torch.mul((fg - fd), fm)

class asff_affusion(nn.Module):
    """Some Information about asff_affusion"""
    def __init__(self):
        super(asff_affusion, self).__init__()
        self.__1st = asff_block()
        self.__2nd = asff_block()
        self.__3rd = asff_block()
        self.__4st = asff_block()
    def forward(self, degraded_feature, guidance_feature, mask_features):
        degraded_feature = self.__1st(degraded_feature, guidance_feature, mask_features)
        degraded_feature = self.__2nd(degraded_feature, guidance_feature, mask_features)
        degraded_feature = self.__3rd(degraded_feature, guidance_feature, mask_features)
        return self.__4st(degraded_feature, guidance_feature, mask_features)

class interpolation(nn.Module):
    def __init__(self, size = None, scale_factor = None, mode = 'nearest', align_corners = False):
        super(interpolation, self).__init__()
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.__interp = nn.functional.interpolate

    def forward(self, x):
        return self.__interp(x, scale_factor = self.scale_factor, mode = self.mode, align_corners = self.align_corners)

class dilated_residual_block(nn.Module):
    def __init__(self, batch, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, leaky = 0.2):
        super(dilated_residual_block, self).__init__()
        self.__dilated_block = nn.Sequential(
            collections.OrderedDict(
				[
					('layer-00', nn.BatchNorm2d(batch)),
					('layer-01', nn.LeakyReLU(leaky)),
					('layer-02', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=True)),
					('layer-03', nn.BatchNorm2d(batch)),
					('layer-04', nn.LeakyReLU(leaky)),
					('layer-05', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=True)),
				]
			)
        )
    def forward(self, x):
        return x + self.__dilated_block(x)

class adaptive_instance_normalization(nn.Module):
    def __init__(self) -> None:
        super(adaptive_instance_normalization, self).__init__()
    def __calc_mean_std(self, features):
        n, c, h, w = features.size()
        size = h * w
        mean = features.mean(dim = (2, 3), keepdim=True)
        std = (((features - mean) * (features - mean)).mean(dim = (2, 3), keepdim=True) * size).sqrt()
        return mean, std
    def forward(self, _1, _2):
        _1_mean, _1_std = self.__calc_mean_std(_1)
        _2_mean, _2_std = self.__calc_mean_std(_2)
        return torch.add(torch.mul(_1_std, torch.sub(_2, _2_mean)), _2_mean)

class asff_network(nn.Module):
    """Some Information about asff_network"""
    def __init__(self):
        super(asff_network, self).__init__()
        self.__adaptive_instance_normalization = adaptive_instance_normalization()
        self.__asff_affusion = asff_affusion()
        self.__degraded_feature = asff_feature()
        self.__guidance_feature = asff_feature()
        self.__mask_feature = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', nn.Conv2d(1, 64, 9, 2, 4, bias=False)),
                    ('layer-01', nn.Conv2d(64, 64, 3, 1, 1, bias=False)),
                    ('layer-02', nn.Conv2d(64, 64, 7, 1, 3, bias=False)),
                    ('layer-03', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
                    ('layer-04', nn.Conv2d(128, 128, 5, 2, 2, bias=False)),
                    ('layer-05', nn.Conv2d(128, 128, 3, 1, 1, bias=False)),
                    ('layer-06', nn.Conv2d(128, 128, 3, 1, 1, bias=False)),
                    ('layer-07', nn.Conv2d(128, 128, 3, 1, 1, bias=False)),
                    ('layer-08', nn.Conv2d(128, 128, 3, 1, 1, bias=False)),
                    ('layer-09', nn.Conv2d(128, 128, 3, 1, 1, bias=False)),
                ]
            )
        )
        self.__reconstruction = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', nn.Conv2d(128, 256, 3, 1, 1, bias=True)),
                    ('layer-01', dilated_residual_block(256, 256, 256, 3, 1, 1, 1, 0.2)),
                    ('layer-02', dilated_residual_block(256, 256, 256, 3, 1, 1, 1, 0.2)),
                    ('layer-03', interpolation(scale_factor=2, mode='bilinear')),
                    ('layer-04', nn.Conv2d(256, 128, 3, 1, 1, bias=True)),
                    ('layer-05', dilated_residual_block(128, 128, 128, 3, 1, 1, 1, 0.2)),
                    ('layer-06', dilated_residual_block(128, 128, 128, 3, 1, 1, 1, 0.2)),
                    ('layer-07', interpolation(scale_factor=2, mode='bilinear')),
                    ('layer-08', nn.Conv2d(128, 32, 3, 1, 1, bias=True)),
                    ('layer-09', dilated_residual_block(32, 32, 32, 3, 1, 1, 1, 0.2)),
                    ('layer-10', dilated_residual_block(32, 32, 32, 3, 1, 1, 1, 0.2)),
                    ('layer-11', nn.Conv2d(32, 3, 1, 1, 0, bias=True)),
                    ('layer-12', nn.Tanh())
                ]
            )
        )

    def forward(self, degraded_image, guidance_image, mask_image):
        fd = self.__degraded_feature(degraded_image)
        fg = self.__guidance_feature(guidance_image)
        fm = self.__mask_feature(mask_image)
        fg = self.__adaptive_instance_normalization(fd, fg)
        fd = self.__asff_affusion(fd, fg, fm)
        return self.__reconstruction(fd)