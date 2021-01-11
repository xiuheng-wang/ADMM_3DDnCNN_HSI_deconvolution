% Script for getting blurring kernels
%
% Reference: 
% Learning spectral-spatial prior via 3DDnCNN for hyperspectral image deconvolution
% Xiuheng Wang, Jie Chen, C¨¦dric Richard, David Brie
%
% 2019/08
% Implemented by
% Xiuheng Wang
% xiuheng.wang@mail.nwpu.edu.cn

clear;clc;
close all;

folderTest   = 'data/test/';
folderKernel  = 'data/kernels/';

kernel_name = {'g_k','g_k_1', 'g_k_2', 'c_k', 'm_k', 's_k'};
img_blurred = zeros(31, 512, 512);
for i = 1:6
    folderResult = fullfile(strcat( 'data/blurred/', kernel_name{i}, '_blurred/'));
    if ~exist(folderResult,'file')
        mkdir(folderResult);
    end  
    load(fullfile(strcat( folderKernel, kernel_name{i}, '.mat' )));
    if i == 3
        sigma = 0.03;
    else
        sigma = 0.01;
    end

    for j = 1:12
        load(fullfile(strcat( folderTest, int2str(j-1), '.mat' )));
        img_blurred = imfilter(im2double(permute(img, [2, 3, 1])), kernel, 'circular', 'conv');
        img_blurred = permute(img_blurred, [3, 1, 2]) + sigma * ones(size(img));
        save(strcat(folderResult, int2str(j-1), '.mat'), 'img_blurred');
    end
end