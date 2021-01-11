# coding: utf-8
# Script for getting blurring kernels
#
# Reference: 
# Learning spectral-spatial prior via 3DDnCNN for hyperspectral image deconvolution
# Xiuheng Wang, Jie Chen, Cédric Richard, David Brie
#
# 2019/10
# Implemented by
# Xiuheng Wang
# xiuheng.wang@mail.nwpu.edu.cn

import numpy as np
import os
from functions import gaussian_kernel_2d, circle_kernel_2d, square_kernel_2d, motion_kernel_2d, makedir
import scipy.io as scio

kernel_dir = './data/kernels/'
if not os.path.exists(kernel_dir):
    makedir(kernel_dir)

g_k = gaussian_kernel_2d(15, 1.6)
g_k_1 = gaussian_kernel_2d(15, 2.4)
g_k_2 = gaussian_kernel_2d(15, 1.6)
c_k = circle_kernel_2d()
m_k = motion_kernel_2d()
s_k = square_kernel_2d()

scio.savemat(kernel_dir + 'g_k' + '.mat', {'kernel':g_k})
scio.savemat(kernel_dir + 'g_k_1' + '.mat', {'kernel':g_k_1})
scio.savemat(kernel_dir + 'g_k_2' + '.mat', {'kernel':g_k_2})
scio.savemat(kernel_dir + 'c_k' + '.mat', {'kernel':c_k})
scio.savemat(kernel_dir + 'm_k' + '.mat', {'kernel':m_k})
scio.savemat(kernel_dir + 's_k' + '.mat', {'kernel':s_k})
