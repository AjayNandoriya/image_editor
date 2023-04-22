import os
import numpy as np
import cv2
import logging

LOGGER = logging.getLogger(__name__)
from matplotlib import pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots

class MyApp(object):
    def __init__(self) -> None:
        self.ref_img = None
        self.test_img = None
        self.base_img = None
        self.diff_img = None 
        pass
    def run(self):
        # self.diff_img = compare(self.ref_img, self.test_img)
        self.diff_img = segment_graphcut(self.ref_img)
    def plot(self):
        fig, axes = plt.subplots(2,2,sharex=True, sharey=True)
        if self.ref_img is not None: 
            axes[0,0].imshow(self.ref_img)
            axes[0,0].set_title('ref')
        if self.test_img is not None:
            axes[0,1].imshow(self.test_img)
            axes[0,1].set_title('test')
        if self.base_img is not None:
            axes[1,0].imshow(self.base_img)
            axes[1,0].set_title('base')
        if self.diff_img is not None:
            axes[1,1].imshow(self.diff_img)
            axes[1,1].set_title('diff')
        return fig
    
    def plot_plotly(self):
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True)
        imgs, titles = [],[]
        if self.ref_img is not None: 
            fig.add_trace(px.imshow(self.ref_img, title="ref"), row=1, col=1)
        if self.test_img is not None: 
            fig.add_trace(px.imshow(self.test_img, title="test"), row=1, col=1)
        if self.base_img is not None: 
            fig.add_trace(px.imshow(self.base_img, title="base"), row=1, col=1)
        if self.diff_img is not None: 
            fig.add_trace(px.imshow(self.diff_img, title="diff"), row=1, col=1)

        return fig
    
def compare(img1:np.ndarray, img2:np.ndarray)->np.ndarray:
    LOGGER.info(f'got images with size {img1.shape} {img2.shape}')
    diff_img = img1.astype(np.float32) - img2.astype(np.float32)
    ret, th_img = cv2.threshold(diff_img,0,255, cv2.CV_8U)

    LOGGER.info(f'returning result image with {th_img.sum()} count.')
    return th_img

def get_marker(img:np.ndarray)->np.ndarray:
    bg_dist_th, fg_dist_th = 0.5, 0.8
    # Threshold the image to create a binary mask
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Perform morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Perform distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

    # Threshold the distance transform to create markers
    fg_mask = dist_transform>fg_dist_th*dist_transform.max()

    # Perform morphological opening to remove small objects
    closing = cv2.bitwise_not(opening)
    dist_transform_neg = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
    bg_mask = dist_transform_neg > bg_dist_th*dist_transform_neg.max()
    prob_bg_mask = (dist_transform_neg >0) & ~(bg_mask)
    prob_fg_mask = (dist_transform >0) & ~fg_mask
    
    
    init_mask = bg_mask*0 + fg_mask*1 +  2*prob_bg_mask + 3*prob_fg_mask
    init_mask = init_mask.astype(np.uint8)
    return init_mask
    


def segment_graphcut(img:np.ndarray, plot_img:bool=False)->np.ndarray:
    # # Load image
    # img = cv2.imread(r'C:\dev\repos\ImageDecomposition\src\main\resources\sem_images\SRAM_22nm.jpg')

    # # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img

    # Apply Gaussian blur to remove noise
    img = cv2.GaussianBlur(gray, (5, 5), 0)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    # Define the region of interest (ROI) using a rectangle
    # The ROI should contain both foreground and background regions
    # cv2.grabCut(img, bg_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    init_mask = get_marker(img)
     
    # Create a binary mask for the probable foreground region
    prob_fg_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    prob_fg_mask = init_mask.copy()
    # Refine the segmentation using GraphCut algorithm
    cv2.grabCut(np.tile(img[:,:,np.newaxis],[1,1,3]), prob_fg_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)

    if plot_img:
        # Create a binary mask for the final segmentation
        final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        final_mask[(prob_fg_mask == 1) | (prob_fg_mask == 3)] = 255

        # Apply the mask to the original image
        segmented_img = cv2.bitwise_and(img, img, mask=final_mask)

        colormap = np.arange(256, dtype=np.uint8)
        colormap[:4] = [0,3,1,2]
        prob_fg_mask_map = cv2.applyColorMap(prob_fg_mask, colormap)
        init_mask_map = cv2.applyColorMap(init_mask, colormap)
        # Display the result
        fig,axes = plt.subplots(1,3,sharex=True, sharey=True)
        axes[0].imshow(init_mask_map)
        axes[1].imshow(prob_fg_mask_map)
        axes[2].imshow(img)
        plt.show()

    return prob_fg_mask

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pass
