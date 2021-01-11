# coding: utf-8
# Script for testing 3DDnCNN
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

import numpy as np
from model import Net
from dataset import Dataset_cave_test
from save_image import save_image
import time
import argparse
import  os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ===========================================================
# Test settings
# ===========================================================
parser = argparse.ArgumentParser(description='CAVE database superresolution:')
# model storage path
parser.add_argument('--sigma', type=float, default=0.04, help='Set standard deviation of Gaussian noise ')
# model configuration
parser.add_argument('--num_blocks', type=int, default=8, help="Set numbers of 3D residual blocks")
parser.add_argument('--num_kernels', type=int, default=32, help="Set numbers of 3D kernels")
# model storage path
parser.add_argument('--model_path', type=str, default='./models/', help='Set model storage path')
# results storage path
parser.add_argument('--results_path', type=str, default='./denoised_results/', help='Set results storage path')

args = parser.parse_args()

model_path = args.model_path + 'hsidb_epoch500.pkl'
img_path = './data/test'

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

test_data = Dataset_cave_test(img_path, args.sigma)
num = test_data.__len__()
print('the total number of test images is:', num)

results_path = args.results_path
if not os.path.exists(results_path):
    os.makedirs(results_path)

test_loader = DataLoader(dataset=test_data,
                        num_workers=0, 
                        batch_size=1,
                        shuffle=False,
                        pin_memory=True)

for i, (data_hsi,label) in enumerate(test_loader):
    
    data_hsi = data_hsi.to(device=device, dtype=torch.float)
    data_label = label.to(device=device, dtype=torch.float)

    start = time.time()
    # compute output
    output = model(data_hsi) # output: Batch_szie*1*C*W*H
    end = time.time()
    print('The No', str(i), 'image costs', end - start, 'seconds')

    output = output.to('cpu')
    output = output.detach().numpy()
    output = np.squeeze(output) 

    label = label.to('cpu')
    label = label.detach().numpy()
    label = np.squeeze(label)

    save_image(output, label, args.results_path, i)

