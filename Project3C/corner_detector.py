'''
  File name: corner_detector.py
  Author: Rajiv Patel-O'Connor
  Date created: Nov 7, 2017
'''
from skimage import feature
import numpy as np


def corner_detector(img):
  cimg = feature.corner_harris(img)
  # determine threshold and threshold corner_harris
# determine threshold and threshold corner_harris
  mu = np.mean(cimg)
  s = np.std(cimg)
  threshold = mu + 2.75 * s
  cimg = thresholdImage(cimg, threshold)
  
  return cimg



def thresholdImage(img, threshold):
    img[img<threshold] = 0.0
    return img