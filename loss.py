'''
Author: your name
Date: 1970-01-01 00:00:00
LastEditTime: 2021-02-20 16:09:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ASFFNet/loss.py
ssim : https://github.com/jacke121/pytorch-ssim
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections

import resnet50_256 as residual_net

class vgg_encoder(nn.Module):
    def __init__(self):
        super(vgg_encoder, self).__init__()
        network = residual_net.resnet50_256(weights_path='./model/resnet50_256.pth')
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

class flatten(nn.Module):
    def __init__(self):
        super(flatten, self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)

class total_variance(nn.Module):
    """Some Information about total_variance"""
    def __init__(self, weight=1):
        super(total_variance,self).__init__()
        self.__weight = weight

    def __tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.__tensor_size(x[:,:,1:,:])
        count_w = self.__tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.__weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class style_loss(nn.Module):
    """Some Information about style_loss"""
    def __init__(self):
        super(style_loss, self).__init__()
        self.__vgg_encoder = vgg_encoder()
        self.__vgg_encoder.eval()
        self.__loss = torch.nn.L1Loss()

    def __gram(self, image):
        (b, n, h, w) = image.size()
        features = image.view(b, n, w * h)
        features_transpose = features.transpose(1, 2)
        return features.bmm(features_transpose) / (n * h * w)

    def forward(self, image_1, image_2):
        _1_1, _1_2, _1_3, _1_4 = self.__vgg_encoder(image_1)
        _2_1, _2_2, _2_3, _2_4 = self.__vgg_encoder(image_2)
        _1_loss = self.__loss(self.__gram(_1_1), self.__gram(_2_1))
        _2_loss = self.__loss(self.__gram(_1_2), self.__gram(_2_2))
        _3_loss = self.__loss(self.__gram(_1_3), self.__gram(_2_3))
        _4_loss = self.__loss(self.__gram(_1_4), self.__gram(_2_4))
        return _1_loss + _2_loss + _3_loss + _4_loss

class perceptual_loss(nn.Module):
    """Some Information about perceptual_loss"""
    def __init__(self):
        super(perceptual_loss, self).__init__()
        self.__vgg_encoder = vgg_encoder()
        self.__vgg_encoder.eval()
        self.__loss = torch.nn.L1Loss()

    def forward(self, image_1, image_2):
        _1_1, _1_2, _1_3, _1_4 = self.__vgg_encoder(image_1)
        _2_1, _2_2, _2_3, _2_4 = self.__vgg_encoder(image_2)
        return self.__loss(_1_1, _2_1) + self.__loss(_1_2, _2_2) + self.__loss(_1_3, _2_3) + self.__loss(_1_4, _2_4) 

class degraded_operation(nn.Module):
    """
    TODO: to be modified, add noise or blur operation directly?
    degraded_operation need to be a decoder-encoder structure in this section?
    """
    def __init__(self):
        super(degraded_operation, self).__init__()
        self.__degraded_operation = nn.Sequential(
            collections.OrderedDict(
                [
                    ('degraded_operation-00', nn.Conv2d(3, 32, 3, 1, 1, bias=True)),
                    ('degraded_operation-01', nn.Conv2d(32, 64, 3, 1, 1, bias=True)),
                    ('degraded_operation-02', nn.Conv2d(64, 32, 3, 1, 1, bias=True)),
                    ('degraded_operation-03', nn.Conv2d(32, 3, 3, 1, 1, bias=True))
                ]
            )
        )

    def forward(self, image):
        return self.__degraded_operation(image)

class generator_loss(nn.Module):
    """Some Information about generator_loss"""
    def __init__(self):
        super(generator_loss, self).__init__()
        self.__style_loss = style_loss()
        self.__tv_loss = total_variance()
        self.__mse_loss = torch.nn.MSELoss()
        self.__perceptual_loss = perceptual_loss()
        self.__adversarial_loss = torch.nn.BCELoss()
        # self.__adversarial_loss = torch.nn.CrossEntropyLoss()

    def forward(self, degraded_image, generated_image, guidance_image, encodered_image, generated_tag, mask_image):
        degraded_image = torch.mul(degraded_image, mask_image.expand_as(degraded_image))
        guidance_image = torch.mul(guidance_image, mask_image.expand_as(guidance_image))
        generated_image = torch.mul(generated_image, mask_image.expand_as(generated_image))
        encodered_image = torch.mul(encodered_image, mask_image.expand_as(encodered_image))

        tv_loss = self.__tv_loss(generated_image)
        mse_loss = self.__mse_loss(degraded_image, encodered_image)
        style_loss = self.__style_loss(generated_image, guidance_image)
        perceptual_loss = self.__perceptual_loss(degraded_image, generated_image)

        fake_tag = torch.autograd.Variable(torch.Tensor(generated_tag.size(0), 1).fill_(0.0), requires_grad=False).to(generated_image.device)
        real_tag = torch.autograd.Variable(torch.Tensor(generated_tag.size(0), 1).fill_(1.0), requires_grad=False).to(generated_image.device)

        generated_tag[generated_tag < 0.0] = 0.0
        generated_tag[generated_tag > 1.0] = 1.0

        adversarial_loss = self.__adversarial_loss(generated_tag, real_tag)

        return tv_loss + mse_loss + style_loss + perceptual_loss + adversarial_loss

class discriminator(nn.Module):
    """
    TODO: to be modified, add noise or blur operation?
    the 1st output channel number is np.log2(min(image_width, image_height))
    """
    __const_negative_slope = 0.2
    def __init__(self):
        super(discriminator, self).__init__()
        self.__discriminator_operation = nn.Sequential(
            collections.OrderedDict(
                [
                    ('discriminator_operation-00', nn.Conv2d(3, 8, 3, 1, 1, bias=True)),
                    ('discriminator_operation-01', nn.LeakyReLU(0.2, inplace=True)),
                    ('discriminator_operation-02', nn.AvgPool2d(2)),
                    ('discriminator_operation-03', nn.LeakyReLU(0.2, inplace=True)),
                    ('discriminator_operation-04', nn.Conv2d(8, 16, 3, 1, 1, bias=True)),
                    ('discriminator_operation-05', nn.LeakyReLU(0.2, inplace=True)),
                    ('discriminator_operation-06', nn.AvgPool2d(2)),
                    ('discriminator_operation-07', nn.LeakyReLU(0.2, inplace=True)),
                    ('discriminator_operation-08', nn.Conv2d(16, 32, 3, 1, 1, bias=True)),
                    ('discriminator_operation-09', nn.LeakyReLU(0.2, inplace=True)),
                    ('discriminator_operation-10', nn.AvgPool2d(2)),
                    ('discriminator_operation-11', nn.LeakyReLU(0.2, inplace=True)),
                    ('discriminator_operation-12', nn.Conv2d(32, 64, 3, 1, 1, bias=True)),
                    ('discriminator_operation-13', nn.LeakyReLU(0.2, inplace=True)),
                    ('discriminator_operation-15', nn.AdaptiveAvgPool2d(1)),
                    ('discriminator_operation-16', flatten()),
                    ('discriminator_operation-17', nn.Linear(64, 1)),
                ]
            )
        )
        # self.__linear = nn.Linear(64, 1)

    def forward(self, image):
        # output = self.__discriminator_operation(image)
        # output = output.view(output.size(0), -1)
        # return self.__linear(output)
        return self.__discriminator_operation(image)

class discriminator_loss(nn.Module):
    """
    discriminator_loss include 2 kind of discriminator loss
    """
    def __init__(self):
        super(discriminator_loss, self).__init__()
        self.__adversarial_loss = torch.nn.BCELoss()

    def forward(self, generated_image, guidance_image, tag_package, mask_image):

        guidance_image = torch.mul(guidance_image, mask_image.expand_as(guidance_image))
        generated_image = torch.mul(generated_image, mask_image.expand_as(generated_image))
        
        fake_tag = torch.autograd.Variable(torch.Tensor(tag_package[0].size(0), 1).fill_(0.0), requires_grad=False).to(generated_image.device)
        real_tag = torch.autograd.Variable(torch.Tensor(tag_package[0].size(0), 1).fill_(1.0), requires_grad=False).to(generated_image.device)
        
        tag_package[0][tag_package[0] < 0.0] = 0.0
        tag_package[0][tag_package[0] > 1.0] = 1.0
        tag_package[1][tag_package[1] < 0.0] = 0.0
        tag_package[1][tag_package[1] > 1.0] = 1.0

        fake_loss = self.__adversarial_loss(tag_package[0], fake_tag)
        real_loss = self.__adversarial_loss(tag_package[1], real_tag)
        return fake_loss + real_loss
