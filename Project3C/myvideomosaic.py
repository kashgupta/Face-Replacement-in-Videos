#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 23:31:48 2017

@author: rajivpatel-oconnor
"""
import numpy as np
import imageio
#import matplotlib.pyplot as plt


def myvideomosaic(img_mosaic):
    maxWidth = 0
    maxHeight = 0
    
    for i in range(len(img_mosaic)):
        [height, width, depth] = img_mosaic[i].shape
        if width > maxWidth: maxWidth = width
        if height > maxHeight: maxHeight = height
        
    num_images = len(img_mosaic)
    remainder_h = maxHeight % 16
    remainder_w = maxWidth % 16
    maxHeight += 16 - remainder_h
    maxWidth += 16 - remainder_w
    video=np.zeros((num_images,maxHeight,maxWidth,3),dtype=np.uint8)
    
    #Add original image to front of video
    for i in range(num_images):
        currToPad = np.zeros((maxHeight, maxWidth, 3),dtype=np.uint8)
        current_image = img_mosaic[i]
        currToPad[:current_image.shape[0], :current_image.shape[1]] = current_image
        #plt.imsave('./framesForVideo/' + str(i) + '.png', currToPad)
        video[i, :, :, :]=currToPad
    imageio.mimwrite('video_mosaic_grasp_old_full.mp4', video, fps=15)
