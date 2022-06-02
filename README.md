# Learning Spectral-Spatial Prior via 3DDnCNN for Hyperspectral Image Deconvolution

Steps:

1. Run cave_processing.py to process the public dataset CAVE;

2. Run get_kernel.py to get serveral blurring kernels used in the paper;

3. Run blurring_image.m to blurring the raw hyperspectral images with obtained kernels;

4. Run main_con.py is the main function for the hyperspectral image deconvolution.

If you want to train and test the denoising neural network 3DDnCNN:

1. Run train.py to train the 3DDnCNN; 

2. Run test.py to test the 3DDnCNN;

For any questions, feel free to email me at xiuheng.wang@mail.nwpu.edu.cn.

If this code is helpful for you, please cite our paper as follows:

    @inproceedings{wang2020learning,
      title={Learning Spectral-Spatial Prior Via 3DDNCNN for Hyperspectral Image Deconvolution},
      author={Wang, Xiuheng and Chen, Jie and Richard, C{\'e}dric and Brie, David},
      booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      pages={2403--2407},
      year={2020},
      organization={IEEE}
    }
