import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def create_img(H,W):
    P = H//8
    C = P//2
    img = np.zeros([H,W], dtype=np.float32)
    for y1 in range(P//4, H, P):
        y2 = min(y1+C, H)
        img[y1:y2,:] = 1.0
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
    filtered_img = np.zeros((height, width), dtype=img_padded.dtype)
    half_window = window_size // 2

    
    spatial_weights = gaussian_kernel(window_size, sigma_s)
    for i0 in range(height):
        for i1 in range(width):
            # Calculate spatial weights within the window

            img_clip = img_padded[i0:i0 + window_size, i1:i1 + window_size]
            
            # Calculate range weights within the window
            intensity_diff = img_clip - img_padded[i0 + half_window, i1 + half_window]  
            range_weights = np.exp(-intensity_diff**2 / (2 * sigma_r**2))

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
    filtered_img = np.zeros((height, width), dtype=img_padded.dtype)
    half_window = window_size // 2

    # img_padded = np.pad(img, half_window, mode='symmetric')

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
    H,W = 128, 128
    NC = 2

    # filter
    window_size = 5
    sigma_s = 1
    sigma_r = 5*10
    sigma_r1 = 10*(25-5)/(5*255)

    half_window = window_size//2
    img = create_img(H,W)*255

    img = cv2.GaussianBlur(img, (7,7), 0)
    imgs = []
    fimgs1 = []
    fimgs2 = []
    for i in range(NC):
        nimg = add_noise(img, 5, 15)
        imgs.append(nimg)
        img_padded = np.pad(nimg, half_window, mode='symmetric')

        fimg1 = bilateral_filter_windowed(img_padded, sigma_s, sigma_r, window_size)
        fimgs1.append(fimg1)
        fimg2 = bilateral_scaled_filter_windowed(img_padded, sigma_s, sigma_r, sigma_r1, window_size)
        fimgs2.append(fimg2)


    fig,axs = plt.subplots(3,3,sharex=True, sharey=True)
    vmax = 40
    for ic in range(NC):
        axs[0,ic].imshow(imgs[ic])
    axs[0,2].imshow(imgs[1]-imgs[0], vmin=-vmax, vmax=vmax)
    for ic in range(NC):
        axs[1,ic].imshow(fimgs1[ic])
    axs[1,2].imshow(fimgs1[1]-fimgs1[0], vmin=-vmax, vmax=vmax)
    for ic in range(NC):
        axs[2,ic].imshow(fimgs2[ic])
    axs[2,2].imshow(fimgs2[1]-fimgs2[0], vmin=-vmax, vmax=vmax)
    
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