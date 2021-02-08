""" © 2018, lizhengwei """
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from PIL import Image, ImageOps
import torchvision

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class MS_SSIM(torch.nn.Module):
    def __init__(self, device, size_average = True, max_val = 255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val
        self.device = device

    def _ssim(self, img1, img2, size_average = True):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).to(self.device)
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def l1_oss(self, img1, img2):
        _, c, w, h = img1.size()
        window_size = min(w, h, 7)
        sigma = 1.1 * window_size / 7
        window = create_window(window_size, sigma, self.channel).to(self.device)

        imgdiff = torch.abs(img1 - img2)
        mu1 = F.conv2d(imgdiff, window, padding=window_size // 2, groups=self.channel)

        return mu1.mean()

    def clr_loss(self, img1, img3):
        _, c, w, h = img1.size()
        window_size = min(w, h, 29)
        sigma = window_size / 6
        window = create_window(window_size, sigma, self.channel).to(self.device)
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
        mu3 = F.conv2d(img3, window, padding=window_size // 2, groups=self.channel)

        weigray = torch.tensor([0.299, 0.587, 0.144])
        weigray = weigray.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)

        c_wei = [-0.000519, 0.076330, -0.173433, 1.435936, -1.894404, 1.556091]

        mu1 = c_wei[5]*mu1.pow(5) + c_wei[4]*mu1.pow(4) + c_wei[3]*mu1.pow(3) + c_wei[2]*mu1.pow(2) + c_wei[1]*mu1 + c_wei[0]
        #mu3 = c_wei[5]*img3.pow(5) + c_wei[4]*img3.pow(4) + c_wei[3]*img3.pow(3) + c_wei[2]*img3.pow(2) + c_wei[1]*img3 + c_wei[0]

        mug1 = F.conv2d(mu1, weigray, padding=0)
        mug3 = F.conv2d(mu3, weigray, padding=0)

        me1 = mug1.mean(2, keepdim=True).mean(3, keepdim=True)
        me3 = mug3.mean(2, keepdim=True).mean(3, keepdim=True)

        sc = torch.div(me1, me3)
        mu3 = mu3 * sc

        #rat = torch.div(mug1, mug3+0.01).clamp(0.125, 8)
        #rat = torch.cat((rat, rat, rat), 1)
        #mu3 = mu3 * rat

        #loss = torch.nn.MSELoss()
        #output = loss(mu1, mu3)
        #return  output

        imgdiff = torch.abs(mu1 - mu3)
        return imgdiff.mean()

    def ms_ssim(self, img1, img2, levels=5):
        _, c, w, h = img1.size()
        self.channel = c

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(self.device))

        msssim = Variable(torch.Tensor(levels,).to(self.device))
        mcs = Variable(torch.Tensor(levels,).to(self.device))
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1]**weight[0:levels-1])*(msssim[levels-1]**weight[levels-1]))
        value = 1 - value

        return value

    def forward(self, img1, img2, img3):
        wei_0 = 0.4
        wei_1 = 0.4
        wei_2 = 0.2

        l1v = self.l1_oss(img1, img2)
        clrv = self.clr_loss(img1, img3)

        ssimv = self.ms_ssim(img1, img2)

        value = wei_0 * ssimv + wei_1 * l1v + wei_2 * clrv

        return value
