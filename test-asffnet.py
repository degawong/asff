'''
Author: your name
Date: 2021-01-04 09:19:42
LastEditTime: 2021-01-15 02:32:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \ASFFNet\test.py
'''

import os
import argparse
from PIL import Image
import torch
import torchvision
import onnxruntime as ort
from torchvision import transforms
from network import ASFFNet
from dataset import Data, ASFFTest
from PIL import Image, ImageOps
import numpy as np

device = 'cuda'

# TODO : save original input data
pb_path = "checkpoint/pb/025_epoch.pb"
onnx_path = "checkpoint/onnx/025_epoch.onnx"
pytorch_path = "checkpoint/model/025_epoch.pth"
# a = np.random.normal(0,1,(2,2),dtype=np.float32)
# a.tofile('a.bin')
# np.fromfile('xxx', dtype=np.uint8)
# features_mean.cpu().detach().numpy().tofile('data/mean_1.txt',sep=" ",format="%s")
def onnx_to_pb():
    """Some Information about onnx_to_pb"""
    tf_exp = onnx_tf.backend.prepare(onnx.load(onnx_path))
    tf_exp.export_graph(pb_path)

def pytorch_to_onnx(network):
    input_names = [
        "degraded_image", 
        "guidance_image", 
        "mask_image", 
    ]
    degraded_image = torch.randn(
        [1, 3, 256, 256]
    )
    guidance_image = torch.randn(
        [1, 3, 256, 256]
    )
    mask_image = torch.randn(
        [1, 1, 256, 256]
    )
    output_names = ["output"]
    torch.onnx.export(
        network, 
        (degraded_image.to(device), guidance_image.to(device), mask_image.to(device)),
        onnx_path, 
        verbose=True, 
        input_names = input_names,
        output_names = output_names
    )

def test_process():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--degraded_directory', type=str, default='test/degraded_image',
                        help='degraded image directory for train')
    parser.add_argument('--guidance_directory', type=str, default='test/guidance_image',
                        help='guidance image directory for train')
    parser.add_argument('--mask_driectory', type=str, default='test/mask_image',
                        help='mask image directory for train')
    parser.add_argument('--gpu', '-g', type=int, default=6,
                        help='GPU ID(nagative value indicate CPU)')
    args = parser.parse_args()

    network = ASFFNet()
    network.load_state_dict(torch.load(pytorch_path))
    network.eval()
    network = network.to(device)
    pytorch_to_onnx(network)
    data = torch.utils.data.DataLoader(
        ASFFTest(
            args.degraded_directory, 
            args.guidance_directory, 
            args.mask_driectory,
        )
    )
    for i, (degraded_image, guidance_image, mask_image, degraded_path) in enumerate(data, 0):
        with torch.no_grad():
            pytorch_image = network(
                    degraded_image.to(device), 
                    guidance_image.to(device), 
                    mask_image.to(device)
            )
        torchvision.utils.save_image(pytorch_image, degraded_path[0].replace('degraded_image', 'pytorch'))
        onnx_image = ort.InferenceSession(onnx_path).run(
            None,
            {
                'degraded_image': degraded_image.cpu().numpy(),
                'guidance_image': guidance_image.cpu().numpy(),
                'mask_image': mask_image.cpu().numpy()
            }
        )
        torchvision.utils.save_image(torch.Tensor(onnx_image).squeeze(0).to(device), degraded_path[0].replace('degraded_image', 'onnx'))

# for pytorch version 1.3 or older, the torch.Tensor function is different from privious
if __name__ == '__main__':
    test_process()
