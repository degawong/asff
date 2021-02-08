'''
Author: your name
Date: 1970-01-01 00:00:00
LastEditTime: 2021-01-12 02:28:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ASFFNet/dataset.py
'''
import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image, ImageOps

import os
import re
import cv2
import imageio
import numpy as np
import scipy as sp
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc, imread, imwrite, resize

class Data():
    """
    Data
    """
    def walk_path(self, path, experession=".*\.(bmp|jpg|png|BMP|JPG|PNG)"):
        assert os.path.exists(path), "directory {} does not exist".format(path)
        return [os.path.join(path, _) for _ in os.listdir(path) if re.match(experession, _)]
    def make_gif(self, image_path, name, duration = 0.50):
        frames = [imread(_) for _ in self.walk_path(image_path)]
        imageio.mimsave(name, frames, 'gif', duration=duration)
    def image2video(self, image_path, name, fps = 25):
        image_list = self.walk_path(image_path)
        # VideoWriter_fourcc为视频编解码器 ('I', '4', '2', '0') —>(.avi) 、('P', 'I', 'M', 'I')—>(.avi)、('X', 'V', 'I', 'D')—>(.avi)、('T', 'H', 'E', 'O')—>.ogv、('F', 'L', 'V', '1')—>.flv、('m', 'p', '4', 'v')—>.mp4
        fourcc = VideoWriter_fourcc(*"MP4V")
        videoWriter = cv2.VideoWriter(name, fourcc, fps, Image.open(image_list[0]).size)
        for _ in image_list:
            videoWriter.write(cv2.imread(_))
    def video2image(self, video_path, image_directory, fps = 25):
        index = -1
        cap = cv2.VideoCapture(video_path)
        _ = cap.isOpened()
        while _:
            index += 1
            _, image = cap.read()
            imwrite('{}/{:08d}.bmp'.format(image_directory, index), image)
        cap.release()
        
class ASFFTest(Data, Dataset):
    """
    ASFF dataset process class
    """
    def __init__(self, degraded_directory, guidance_directory, mask_directory):
        degraded_package = self.walk_path(degraded_directory)
        guidance_package = self.walk_path(guidance_directory)
        mask_package = self.walk_path(mask_directory)
        self.__transforms = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.__tensor_rgb2gray = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels = 1),
                transforms.ToTensor(),
            ]
        )        
        self.__list = list(zip(degraded_package, guidance_package, mask_package))

    def __len__(self):
        return len(self.__list)

    def __getitem__(self, index):
        [degraded_path, guidance_path, mask_path] = self.__list[index]
        degraded_image = self.__transforms(Image.open(degraded_path))
        guidance_image = self.__transforms(Image.open(guidance_path))
        mask_image = self.__tensor_rgb2gray(Image.open(mask_path))
        return [degraded_image, guidance_image, mask_image, degraded_path]

class ASFFDataSet(Data, Dataset):
    """
    ASFF dataset process class
    """
    def __init__(self, degraded_directory, guidance_directory, mask_directory, albedo_directory, groundtruth_directory):
        self.__transforms = transforms.Compose([
                                transforms.ToTensor()
                            ])
        degraded_package = self.walk_path(degraded_directory)
        guidance_package = self.walk_path(guidance_directory)
        mask_package = self.walk_path(mask_directory)
        albedo_package = self.walk_path(albedo_directory)
        groundtruth_package = self.walk_path(groundtruth_directory)
        self.__list = list(zip(degraded_package, guidance_package, mask_package, albedo_package, groundtruth_package))
        np.random.shuffle(self.__list)

    def __len__(self):
        return len(self.__list)

    def __getitem__(self, index):
        [degraded_path, guidance_path, mask_path, albedo_path, groundtruth_path] = self.__list[index]        
        degraded_image = self.__transforms(Image.open(degraded_path))
        guidance_image = self.__transforms(Image.open(guidance_path))
        mask_image = self.__transforms(Image.open(mask_path))
        albedo_image = self.__transforms(Image.open(albedo_path))
        groundtruth_image = self.__transforms(Image.open(groundtruth_path))
        return [degraded_image, guidance_image, mask_image, albedo_image, groundtruth_image]

class Draw():
    """Some Information about Draw"""
    def __init__(self):
        pass
    def scatter(self, x, y, color = 'b', marker = 'o', annotation = False):
        assert(len(x) == len(y), "input size of x &&y do not match")
        figure = plt.figure().add_subplot(111)
        figure.scatter(x, y, c = color, marker = marker)
        if annotation is True:
            for i, txt in enumerate(range(len(x)), 0):
                figure.annotate(txt, (x[i], x[i]))
            plt.axis('off')
            plt.show()
    def scatter(self, point_path, spliter = ' ', color = 'b', marker = 'o', annotation = False):
        figure = plt.figure().add_subplot(111)
        x, y = [], []
        for line in open(point_path, 'r'):
            information = [(s) for s in line.split(spliter)]
            x.append(float(information[0]))
            y.append(float(information[1]))
        figure.scatter(x, y, c = color, marker = marker)
        if annotation is True:
            for i, txt in enumerate(range(len(x)), 0):
                figure.annotate(txt, (x[i], y[i]))
            plt.axis('off')
            plt.show()
    def plot(self, x, y, marker = 'o', xlabel = 'x', ylabel = 'y', label = 'small name', title = 'loss curve', rotation = 0):
        plt.plot(x, y, marker = marker, label = label)
        plt.legend()
        # plt.xticks(x, names, rotation = rotation)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()



