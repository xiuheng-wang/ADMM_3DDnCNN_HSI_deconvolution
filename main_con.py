# coding: utf-8
# Script for performing hyperspectral image deconvolution 
#
# Reference: 
# Learning spectral-spatial prior via 3DDnCNN for hyperspectral image deconvolution
# Xiuheng Wang, Jie Chen, Cédric Richard, David Brie
#
# 2019/10
# Implemented by
# Xiuheng Wang
# xiuheng.wang@mail.nwpu.edu.cn

from __future__ import division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net
import numpy as np
from save_image import save_image
import time
import tifffile as tiff
import cv2
from scipy.fftpack import fft2, ifft2 
import matplotlib.pyplot as plt
from functions import gaussian_kernel_2d, circle_kernel_2d, square_kernel_2d, motion_kernel_2d, makedir, psf2otf, get_blurred, center_crop
from save_image import save_image, psnr, mse
import  os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import scipy.io as scio
import argparse

# ===========================================================
# settings
# ===========================================================
parser = argparse.ArgumentParser(description='CAVE database deblurring:')
# model storage path
parser.add_argument('--model_path', type=str, default='./models/', help='Set model storage path')
# bluuring images path
parser.add_argument('--inputs_path', type=str, default='/g_k_blurred/', help='Set results storage path')
# results storage path
parser.add_argument('--results_path', type=str, default='/g_k_deblurred/', help='Set results storage path')

args = parser.parse_args()

raw_image_dir = './data/test/'
blurred_image_dir = './data/blurred' + args.inputs_path
deblurred_image_dir = './data/deblurred' + args.results_path
num_images = 12
model_path = args.model_path + 'hsidb_epoch500.pkl'

# select blur kernel and hyperparameters
kernel = gaussian_kernel_2d()
rho = 0.06  # set rho = 0.8 for the scenario (c) due to its different noise level
Iteration = 20

# Initialize denoise nerual network: 1 --> GPU mode, 0 --> Cpu mode
# It is strongly recommended to use GPU mode as its speed is extremely faster than CPU mode.
# If your GPU memory is limited, we recommend to crop the input of the 3DDnCNN into serveral pathes
# and integrate them after denoising.
mode = 1
if mode:
##### GPU mode #####
    device="cuda:0"
    model = Net(num_blocks=8, num_kernels=32)
    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    model = nn.DataParallel(model)
    model.load_state_dict(model_param)
else:
##### CPU mode #####
    device="cpu"
    model = Net(num_blocks=8, num_kernels=32)
    pretrain = torch.load(model_path, map_location=lambda storage, loc: storage)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in pretrain.items():
        if k=='state_dict':
            state_dict=OrderedDict()
            for keys in v:
                name = keys[7:]# remove `module.`
                state_dict[name] = v[keys]
                new_state_dict[k]=state_dict
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict['state_dict'])

# convert model to device
model=model.to(device=device, dtype=torch.float)
model.eval()

if __name__ == '__main__':

    if not os.path.exists(deblurred_image_dir):
        makedir(deblurred_image_dir)

    # for j in range(8):
    plt.figure(0)
    MSE = np.zeros([num_images, Iteration + 1])

    for i in range(num_images):
        img = scio.loadmat(raw_image_dir + str(i) + '.mat')['img']
        # img = tiff.imread(raw_image_dir + str(i) + '.tif')
        dim = np.shape(img)

        # getting blurred images
        img_blurred = scio.loadmat(blurred_image_dir + str(i) + '.mat')['img_blurred']
        img_blurred = np.clip(img_blurred, 0, 1)
        # img_blurred = get_blurred(img, kernel, noise_sigma)
        # tiff.imsave(blurred_image_dir + str(i) + '.tif', (img_blurred * 255).astype(np.uint8))

        # deblurring these images
        height = dim[1]
        width = dim[2]
        # img = center_crop(img, [height, width])
        H = psf2otf(kernel, [height, width])
        HTH = abs(H) ** 2
        HTH = np.tile(np.expand_dims(HTH, 0), [dim[0],1,1])

        img = img.astype(np.float32) / 255 # Normalized
        img_cc = center_crop(img, [498, 498])
        img_blurred_cc = center_crop(img_blurred, [498, 498])
        print('deblurring No ' + str(i) + ' image...')
        print('before deblurring: ', 'MSE: ', mse(img_cc, img_blurred_cc), 'PSNR: ', psnr(img_cc, img_blurred_cc))

        HT = np.tile(np.expand_dims(H.conjugate(), 0), [dim[0],1,1])
        HTY = HT * fft2(img_blurred)
        MSE[i, 0] = mse(img_cc, img_blurred_cc)

        # ADMM
        # Initialize variables
        x = img_blurred
        z = x
        u = np.zeros(dim).astype(np.float32)
        for iter in range(Iteration):
            x_tilde = z - u
            x = np.real(ifft2((HTY + rho * fft2(x_tilde)) / (HTH + rho)))
            z_tilde = x + u

            z_tilde = torch.from_numpy(z_tilde).unsqueeze(0).unsqueeze(0)
            z_tilde = z_tilde.to(device=device, dtype=torch.float)
            z = model(z_tilde) # Please crop z_tilde into serveral patches if the GPU memory is limited.
            z = np.squeeze(z.to('cpu').detach().numpy())

            u = u + x - z
            
            z_cc = center_crop(z, [498, 498])
            MSE[i, iter+1] = mse(z_cc, img_cc)

        save_image(z_cc, img_cc, deblurred_image_dir, i)
        print('after deblurring: ', 'MSE: ', mse(z_cc, img_cc), 'PSNR: ', psnr(z_cc,img_cc))
        print('Done')
    #     plt.plot(MSE[i])

    # scio.savemat(deblurred_image_dir + 'RMSE.mat', {'RMSE':MSE})
    # plt.savefig(deblurred_image_dir + 'RMSE.png')
    # plt.show()
