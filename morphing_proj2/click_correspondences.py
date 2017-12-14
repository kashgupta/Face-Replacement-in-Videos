'''
    File name: click_correspondences.py
    Author: Rajiv Patel-O'Connor
    Date created: 10-18-2017
    '''

'''
    File clarification:
    Click correspondences between two images
    - Input im1: target image
    - Input im2: source image
    - Output im1_pts: correspondences coordiantes in the target image
    - Output im2_pts: correspondences coordiantes in the source image
    '''

import numpy as np
import matplotlib.pyplot as plt
from cpselect import cpselect


def click_correspondences(im1, im2):

    # Display images and Select
    im1_clicked_pts, im2_clicked_pts = cpselect(im1, im2)
    
    # add boundary points
    h, w, d = np.shape(im1)
    upper_left = [[0, 0]]
    mid_left = [[0, np.floor(h/2)]]
    lower_left = [[0, h - 1]]
    upper_mid = [[np.floor(w/2), 0]]
    lower_mid = [[np.floor(w/2), h]]
    upper_right = [[w - 1, 0]]
    mid_right = [[w - 1, np.floor(h / 2)]]
    lower_right = [[w - 1, h - 1]]
    
    im1_pts = np.concatenate(
                             (im1_clicked_pts, upper_left, mid_left, lower_left, upper_mid, lower_mid, upper_right, mid_right,
                              lower_right), axis=0)
    im2_pts = np.concatenate(
                             (im2_clicked_pts, upper_left, mid_left, lower_left, upper_mid, lower_mid, upper_right, mid_right,
                              lower_right), axis=0)

    return im1_pts, im2_pts
