import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def create_img(H,W, P=32, C=16):
    # P = 32 #H//8
    # C = P//2
    img = np.zeros([H,W], dtype=np.float32)
    Ci = int(C)
    Cd= C-Ci
    for y1 in range(P//4, H, P):
        y2 = min(y1+Ci, H)
        img[y1:y2,:] = 1.0
        if y2<H and Cd>0:
            img[y2+1,:] = Cd
    return img

def add_noise(img, offset_sigma:float=0.1, scale_sigma:float=0.0):
    norm_nimg = np.random.normal(0, offset_sigma, img.shape).astype(np.float32)

    weight = (img/255.0)*(scale_sigma-offset_sigma)/offset_sigma + 1
    nimg = norm_nimg*weight
    out_img= img + nimg
    return out_img


import numpy as np
from numba import jit

import numpy as np
from numba import jit

@jit
def gaussian_kernel(size:int, sigma:float):
  """
  Creates a 2D Gaussian kernel.

  Args:
    size: The size of the kernel (e.g., 5x5).
    sigma: The standard deviation of the Gaussian distribution.

  Returns:
    A 2D NumPy array representing the Gaussian kernel.
  """

  kernel_size = size // 2
  kernel = np.empty(shape=(size, size), dtype=np.float32)
  for i0 in range(-kernel_size, kernel_size+1):
    for i1 in range(-kernel_size, kernel_size+1):
      kernel[i0,i1] = np.exp(-(i0**2 + i1**2) / (2 * sigma**2))
  return kernel / np.sum(kernel)

def test_gaussian_kernel():
    kernel = gaussian_kernel(size=5, sigma=0.6)
    pass

@jit
def gaussian_filter_windowed(img_padded:np.ndarray, sigma_s:float, window_size:int):
    """
    Applies a bilateral filter to the input image with a windowed kernel.

    Args:
        img: The input image as a NumPy array.
        sigma_s: Standard deviation for spatial kernel.
        sigma_r: Standard deviation for range kernel.
        window_size: Size of the square window for filtering.

    Returns:
        The filtered image.
    """

    height, width = img_padded.shape[:2]
    height, width = height - window_size + 1, width - window_size + 1
    filtered_img = np.empty((height, width), dtype=img_padded.dtype)
    half_window = window_size // 2

    
    spatial_weights = gaussian_kernel(window_size, sigma_s)
    weights = spatial_weights
    
    for i0 in range(height):
        for i1 in range(width):
            # Calculate spatial weights within the window

            img_clip = img_padded[i0:i0 + window_size, i1:i1 + window_size]
            
            filtered_img[i0, i1] = np.sum(weights * img_clip)

    return filtered_img

@jit
def gaussian_unfilter_windowed(img_padded:np.ndarray, sigma_s:float, window_size:int):
    """
    Applies a bilateral filter to the input image with a windowed kernel.

    Args:
        img: The input image as a NumPy array.
        sigma_s: Standard deviation for spatial kernel.
        sigma_r: Standard deviation for range kernel.
        window_size: Size of the square window for filtering.

    Returns:
        The filtered image.
    """

    height, width = img_padded.shape[:2]
    height, width = height - window_size + 1, width - window_size + 1
    filtered_img = np.empty((height, width), dtype=img_padded.dtype)
    half_window = window_size // 2

    
    spatial_weights = gaussian_kernel(window_size, sigma_s)
    weights = spatial_weights
    weights /= np.sum(weights)
    
    w = weights[half_window,half_window]
    weights *= -1
    weights[half_window,half_window] = 1.0 + 1 - w

    for i0 in range(height):
        for i1 in range(width):
            # Calculate spatial weights within the window

            img_clip = img_padded[i0:i0 + window_size, i1:i1 + window_size]
            
            filtered_img[i0, i1] = np.sum(weights * img_clip)

    return filtered_img

@jit
def bilateral_filter_windowed(img_padded:np.ndarray, sigma_s:float, sigma_r:float, window_size:int):
    """
    Applies a bilateral filter to the input image with a windowed kernel.

    Args:
        img: The input image as a NumPy array.
        sigma_s: Standard deviation for spatial kernel.
        sigma_r: Standard deviation for range kernel.
        window_size: Size of the square window for filtering.

    Returns:
        The filtered image.
    """

    height, width = img_padded.shape[:2]
    height, width = height - window_size + 1, width - window_size + 1
    filtered_img = np.empty((height, width), dtype=img_padded.dtype)
    half_window = window_size // 2

    
    spatial_weights = gaussian_kernel(window_size, sigma_s)
    for i0 in range(height):
        for i1 in range(width):
            # Calculate spatial weights within the window

            img_clip = img_padded[i0:i0 + window_size, i1:i1 + window_size]
            
            # Calculate range weights within the window
            intensity_diff = img_clip - img_padded[i0 + half_window, i1 + half_window]  
            range_weights = np.exp(-intensity_diff**2 / (2 * sigma_r**2))
            range_weights /= np.sum(range_weights) 
            
            weights = spatial_weights * range_weights
            weights /= np.sum(weights)

            filtered_img[i0, i1] = np.sum(weights * img_clip)

    return filtered_img

@jit
def bilateral_scaled_filter_windowed(img_padded:np.ndarray, sigma_s:float, sigma_r0:float,sigma_r1:float, window_size:int):
    """
    Applies a bilateral filter to the input image with a windowed kernel.

    Args:
        img: The input image as a NumPy array.
        sigma_s: Standard deviation for spatial kernel.
        sigma_r0: Standard deviation for range kernel-0.
        sigma_r1: Standard deviation for range kernel-1.
        window_size: Size of the square window for filtering.

    Returns:
        The filtered image.
    """

    height, width = img_padded.shape[:2]
    height, width = height-window_size+1, width -window_size+1
    filtered_img = np.empty((height, width), dtype=img_padded.dtype)
    half_window = window_size // 2

    spatial_weights = gaussian_kernel(window_size, sigma_s)
    for i0 in range(height):
        for i1 in range(width):
            # Calculate spatial weights within the window

            img_clip = img_padded[i0:i0 + window_size, i1:i1 + window_size]
            ref_intensity = img_padded[i0+half_window, i1+half_window]
            # Calculate range weights within the window
            intensity_diff = img_clip - ref_intensity  
            sigma_r = sigma_r0 + sigma_r1*ref_intensity
            range_weights = np.exp(-intensity_diff**2 / (2 * sigma_r**2))
            range_weights /= np.sum(range_weights) 
            weights = spatial_weights * range_weights
            weights /= np.sum(weights)

            filtered_img[i0, i1] = np.sum(weights * img_clip)

    return filtered_img

def filter_noise(img):

    sigma_s = 0.6
    sigma_r = 5
    window_size = 5

    fimg1 = bilateral_filter_windowed(img, sigma_s, sigma_r, window_size)

    fimg2 =bilateral_scaled_filter_windowed(img, sigma_s, sigma_r0=sigma_r, sigma_r1=sigma_r1, window_size=window_size)
    return img

def test_add_noise_filter():
    H,W = 64, 64
    NC = 2

    # filter
    window_size = 7
    sigma_s = 1
    sigma_r = 5
    sigma_r1 = (15-5)/(255)

    half_window = window_size//2
    imgs = []
    nimgs = []
    fimgs1 = []
    fimgs2 = []
    fimgs3 = []
    fimgs4 = []
    for i in range(NC):
        img = create_img(H,W, C=16+i*0.2)*255
        img = cv2.GaussianBlur(img, (7,7), 0)
        imgs.append(img)
        nimg = add_noise(img, 5, 15)
        nimgs.append(nimg)
        img_padded = np.pad(nimg, half_window, mode='symmetric')

        fimg1 = bilateral_filter_windowed(img_padded, 10, sigma_r, window_size)
        fimgs1.append(fimg1)
        fimg2 = bilateral_scaled_filter_windowed(img_padded, 10, sigma_r, sigma_r1, window_size)
        fimgs2.append(fimg2)
        fimg3 = gaussian_filter_windowed(img_padded, sigma_s, window_size)
        fimgs3.append(fimg3)
        fimg3_padded = np.pad(fimg3, half_window, mode='symmetric')
        fimg4 = gaussian_unfilter_windowed(fimg3_padded, sigma_s, window_size)
        fimgs4.append(fimg4)

    data = {'ori':imgs,
            'noise':nimgs,
            'gaussian':fimgs3,
            # 'gaussian2':fimgs4,
            'bilateral':fimgs1,
            'bilateral-1':fimgs2}

    fig,axs = plt.subplots(len(data)+1,4,sharex=True, sharey=True)
    fig, axs2 = plt.subplots(2,1, sharex=True)
    vmax = 40

    for ikey, key in enumerate(data.keys()):
        diff_img = data[key][1]- data[key][0]
        avg_img = (data[key][1]+data[key][0])/(2*255) + 1
        
        for ic in range(NC):
            axs[ikey,ic].imshow(data[key][ic])
            axs[ikey,ic].set_title(f'{key}-{ic}')
        axs[ikey,2].imshow(diff_img, vmin=-vmax, vmax=vmax)
        axs[ikey,2].set_title(f'{key} diff')
        axs[ikey,3].imshow(diff_img/avg_img, vmin=-vmax, vmax=vmax)
        axs[ikey,3].set_title(f'{key} diff norm')

        # axs2[0].plot(np.mean(diff_img,axis=1), label=key)
        # axs2[1].plot(np.mean(diff_img/avg_img,axis=1), label=key)
        axs2[0].plot(diff_img[:,10], label=key)
        axs2[1].plot(diff_img[:,10]/avg_img[:,10], label=key)
        pass
    axs2[0].grid()
    axs2[0].legend()
    axs2[1].grid()
    axs2[1].legend()
    plt.show()
    pass

def test_add_noise():
    H,W = 512, 512
    img = create_img(H,W)*255
    out_img1 = add_noise(img, 5, 25)
    out_img2 = add_noise(img, 5, 25)
    fig,axs = plt.subplots(2,3,sharex=True, sharey=True)
    axs[0,0].imshow(out_img1)
    axs[0,1].imshow(out_img2)
    axs[0,2].imshow(out_img2-out_img1, vmin=-40, vmax=40)
    axs[1,0].imshow(img)
    plt.show()
    pass

if __name__ == '__main__':
    # test_add_noise()
    test_add_noise_filter()
    # test_gaussian_kernel()
    pass