import numpy as np
import cv2 
import os
import math
from scipy.fftpack import fft2, ifft2
from scipy.io import loadmat
import scipy.ndimage

def get_blurred(img, kernel, noise_sigma):
    # get blurred
    img = img.astype(np.float32) / 255
    dim = np.shape(img)
    img_blurred = np.zeros(dim)
    for i in range(dim[0]):
        img_blurred[i, :, :] = scipy.ndimage.convolve(img[i, :, :], kernel)
    # get noised
    img_blurred = img_blurred + noise_sigma * np.random.randn(dim[0], dim[1], dim[2])  # add Gaussian noise
    img_blurred = np.clip(img_blurred, 0, 1)
    return img_blurred

def gaussian_kernel_2d(kernel_size=15, kernel_sigma=1.6):
    kx = cv2.getGaussianKernel(kernel_size,kernel_sigma)
    ky = cv2.getGaussianKernel(kernel_size,kernel_sigma)
    return np.multiply(kx,np.transpose(ky)) 

def circle_kernel_2d(kernel_radius=4):
    kernel_size = 2 * kernel_radius - 1
    kernel = np.zeros([kernel_size, kernel_size])
    x = int((kernel_size-1)/2)
    y = x
    num_ones = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            if np.sqrt((i-x)**2 + (j-y)**2) <= kernel_radius-0.5:
                kernel[i, j] = 1
                num_ones += 1
    return kernel / num_ones

def square_kernel_2d(kernel_sl=5):
    kernel = np.ones([kernel_sl, kernel_sl])
    num_ones = kernel_sl ** 2
    return kernel / num_ones

def motion_kernel_2d():
    kernel = loadmat("./data/Levin09.mat")['kernels'][0, 2]
    # kernel = np.pad(kernel, 3, 'constant')
    return kernel

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def center_crop(img, outsize):
    dim = np.shape(img)[1:3]
    a = int((dim[0]-outsize[0])/2)
    b = int((dim[1]-outsize[1])/2)
    img_cropped = img[:, a:a+outsize[0], b:b+outsize[1]]
    return img_cropped

def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    # otf = np.fft.fft2(psf)
    otf = fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf
