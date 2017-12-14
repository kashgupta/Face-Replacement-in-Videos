#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:10:01 2017

@author: priscillachang
"""

'''
  File name: corner_detector.py
  Author: Rajiv Patel-O'Connor
  Date created: Nov 7, 2017
'''
from skimage import feature
import numpy as np


def corner_detector_fused(img):
  cimg = feature.corner_harris(img)
  # determine threshold and threshold corner_harris
  mu=np.mean(cimg)
  s=np.std(cimg)
  threshold = mu + 2.75 * s
  cimg = thresholdImage(cimg, threshold)
  
  return cimg



def thresholdImage(img, threshold):
    [rows, cols] = img.shape
    for i in range(rows):
        for j in range(cols):
            if img[i][j] < threshold:
                img[i][j] = 0.0
    print(np.nonzero(img)[0].shape)
    return img