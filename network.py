'''
Author: your name
Date: 2021-01-04 17:07:53
LastEditTime: 2021-02-07 16:20:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \ASFFNet\network.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
import numpy as np
import resnet50_256 as VggModel
from ms_ssim_loss import MS_SSIM
from TVLoss import TVLoss
from Myloss import L_color

class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        network = VggModel.resnet50_256(weights_path='./model/resnet50_256.pth')
        network.eval()
        self.slice1 = nn.Sequential(*list(network.children())[:3])
        self.slice2 = nn.Sequential(*list(network.children())[3:7])
        self.slice3 = nn.Sequential(*list(network.children())[7:10])
        self.slice4 = nn.Sequential(*list(network.children())[10:12])
        named_child = list(network.named_children())
        self.slice5 = nn.Sequential()
        self.slice5.add_module(named_child[3][0], named_child[3][1])
        self.slice5.add_module(named_child[12][0], named_child[12][1])
        self.slice5.add_module(named_child[13][0], named_child[13][1])
        self.slice6 = nn.Sequential()
        self.slice6.add_module(named_child[14][0], named_child[14][1])
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h1)
        h6= h4 + h5
        h4 = self.slice6(h6)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4

class ASFFFeature(nn.Module):
    """Some Information about ASFFFeature"""
    def __init__(self):
        super(ASFFFeature, self).__init__()
        self.__Feature = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', nn.Conv2d(3, 64, 3, 1, 1, bias=True)),
                    ('layer-01', DilatedResBlock(64, 64, 64, 3, 1, 7, 7, 0.2)),
                    ('layer-02', DilatedResBlock(64, 64, 64, 3, 1, 5, 5, 0.2)),
                    ('layer-03', nn.Conv2d(64, 128, 3, 2, 1, bias=True)),
                    ('layer-04', DilatedResBlock(128, 128, 128, 3, 1, 5, 5, 0.2)),
                    ('layer-05', DilatedResBlock(128, 128, 128, 3, 1, 3, 3, 0.2)),
                    ('layer-06', nn.Conv2d(128, 128, 3, 2, 1, bias=True)),
                    ('layer-07', DilatedResBlock(128, 128, 128, 3, 1, 3, 3, 0.2)),
                    ('layer-08', DilatedResBlock(128, 128, 128, 3, 1, 1, 1, 0.2)),
                    ('layer-09', nn.Conv2d(128, 128, 3, 1, 1, bias=True)),
                    ('layer-10', nn.LeakyReLU(0.2))
                ]
            )
        )
    def forward(self, x):
        return self.__Feature(x)

class ASFFTanh(nn.Module):
    def __init__(self):
        super(ASFFTanh, self).__init__()
    def forward(self, x):
        return F.tanh(x)

class ASFFBlock(nn.Module):
    """Some Information about ASFFBlock"""
    def __init__(self):
        super(ASFFBlock, self).__init__()
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

class ASFFAFFusion(nn.Module):
    """Some Information about ASFFAFFusion"""
    def __init__(self):
        super(ASFFAFFusion, self).__init__()
        self.__1st = ASFFBlock()
        self.__2nd = ASFFBlock()
        self.__3rd = ASFFBlock()
        self.__4st = ASFFBlock()
    def forward(self, degraded_feature, guidance_feature, mask_features):
        degraded_feature = self.__1st(degraded_feature, guidance_feature, mask_features)
        degraded_feature = self.__2nd(degraded_feature, guidance_feature, mask_features)
        degraded_feature = self.__3rd(degraded_feature, guidance_feature, mask_features)
        return self.__4st(degraded_feature, guidance_feature, mask_features)

class Interpolation(nn.Module):
    def __init__(self, size = None, scale_factor = None, mode = 'nearest', align_corners = False):
        super(Interpolation, self).__init__()
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.__interp = nn.functional.interpolate

    def forward(self, x):
        return self.__interp(x, scale_factor = self.scale_factor, mode = self.mode, align_corners = self.align_corners)

class DilatedResBlock(nn.Module):
    def __init__(self, batch, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, leaky = 0.2):
        super(DilatedResBlock, self).__init__()
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

class AdaIN(nn.Module):
    def __init__(self) -> None:
        super(AdaIN, self).__init__()
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

class ASFFNet(nn.Module):
    """Some Information about ASFFNet"""
    def __init__(self):
        super(ASFFNet, self).__init__()
        self.__loss = Loss()
        self.__adain = AdaIN()
        self.__asff_affusion = ASFFAFFusion()
        self.__DegradedFeature = ASFFFeature()
        self.__GuidanceFeature = ASFFFeature()
        self.__MaskFeature = nn.Sequential(
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
        self.__Reconstruction = nn.Sequential(
            collections.OrderedDict(
                [
                    ('layer-00', nn.Conv2d(128, 256, 3, 1, 1, bias=True)),
                    ('layer-01', DilatedResBlock(256, 256, 256, 3, 1, 1, 1, 0.2)),
                    ('layer-02', DilatedResBlock(256, 256, 256, 3, 1, 1, 1, 0.2)),
                    ('layer-03', Interpolation(scale_factor=2, mode='bilinear')),
                    ('layer-04', nn.Conv2d(256, 128, 3, 1, 1, bias=True)),
                    ('layer-05', DilatedResBlock(128, 128, 128, 3, 1, 1, 1, 0.2)),
                    ('layer-06', DilatedResBlock(128, 128, 128, 3, 1, 1, 1, 0.2)),
                    ('layer-07', Interpolation(scale_factor=2, mode='bilinear')),
                    ('layer-08', nn.Conv2d(128, 32, 3, 1, 1, bias=True)),
                    ('layer-09', DilatedResBlock(32, 32, 32, 3, 1, 1, 1, 0.2)),
                    ('layer-10', DilatedResBlock(32, 32, 32, 3, 1, 1, 1, 0.2)),
                    ('layer-11', nn.Conv2d(32, 3, 1, 1, 0, bias=True)),
                    ('layer-12', ASFFTanh())
                ]
            )
        )
        # only for debug
        # self.__to_image_chcannel = nn.Conv2d(128, 3, 3)

    def forward(self, degraded_image, guidance_image, mask_image):
        fd = self.__DegradedFeature(degraded_image)
        fg = self.__GuidanceFeature(guidance_image)
        fm = self.__MaskFeature(mask_image)
        fg = self.__adain(fd, fg)
        fd = self.__asff_affusion(fd, fg, fm)
        return self.__Reconstruction(fd)

class Loss(nn.Module):
    """Some Information about Loss"""
    def __init__(self):
        super(Loss, self).__init__()
        self.__vgg_encoder = VGGEncoder()
        self.__vgg_encoder.eval()

    def __gram(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def __loss(self, low_image, gt_image, mask_image, alb_image):
        low_image = torch.mul(low_image, mask_image.expand_as(low_image))
        gt_image = torch.mul(gt_image, mask_image.expand_as(gt_image))
        alb_image = torch.mul(alb_image, mask_image.expand_as(alb_image))

        ms_ssim_out = MS_SSIM('cuda', max_val=1)(low_image, gt_image, alb_image)
        loss_mse = ms_ssim_out

        loss_tv = TVLoss()(low_image)
        #loss_clr = self.color_loss(low_image)

        hlow1, hlow2, hlow3, hlow4 = self.__vgg_encoder(low_image)
        hgt1, hgt2, hgt3, hgt4 = self.__vgg_encoder(gt_image)

        loss_fn = torch.nn.L1Loss()

        loss_perc = loss_fn(hlow1, hgt1)
        loss_perc += loss_fn(hlow2, hgt2)
        loss_perc += loss_fn(hlow3, hgt3)
        loss_perc += loss_fn(hlow4, hgt4)
        loss_rec = 3.0 * loss_mse + 0.15 * loss_perc

        gram_low1 = self.__gram(hlow1)
        gram_gt1 = self.__gram(hgt1)
        loss_style = loss_fn(gram_low1, gram_gt1.expand_as(gram_low1))
        gram_low2 = self.__gram(hlow2)
        gram_gt2 = self.__gram(hgt2)
        loss_style += loss_fn(gram_low2, gram_gt2.expand_as(gram_low2))
        gram_low3 = self.__gram(hlow3)
        gram_gt3 = self.__gram(hgt3)
        loss_style += loss_fn(gram_low3, gram_gt3.expand_as(gram_low3))
        gram_low4 = self.__gram(hlow4)
        gram_gt4 = self.__gram(hgt4)
        loss_style += loss_fn(gram_low4, gram_gt4.expand_as(gram_low4))

        loss_real = 0.5e2 * loss_style
        loss_tv = 0.20e2 * loss_tv
        loss = loss_rec + loss_real + loss_tv

        return loss

    def forward(self, image, mask_image, albedo_image, groundtruth_image):
        return self.__loss(image, groundtruth_image, mask_image, albedo_image)
