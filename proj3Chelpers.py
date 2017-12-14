#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:35:15 2017

@author: rajivpatel-oconnor
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from time import gmtime, strftime


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def apply_homography(H, x, y):
    matrix = np.stack((x, y, np.ones(len(x))))
    applied_homography = H.dot(matrix)
    X = np.divide(applied_homography[0], applied_homography[2])
    Y = np.divide(applied_homography[1], applied_homography[2])
    return X, Y

def count_inliers(new_x1, new_y1, x2, y2, threshold):
    inlier_binary = np.zeros((len(new_x1)), dtype=np.bool_)
    count = 0
    for i in range(len(new_x1)):
        if distance(new_y1[i], new_x1[i], y2[i], x2[i]) <= threshold:
            count += 1
            inlier_binary[i] = 1
    return count, inlier_binary

def distance(y1, x1, y2, x2):
    return math.sqrt((x2-x1)**2 + (y2- y1)**2)

def overlay_points(img, x, y, name):
    plt.figure()
    implot = plt.imshow(img)
    plt.scatter(x, y,color='red',marker='o', s=1)
    plt.savefig('postanms' + name)
    plt.close("all")
    
def obtain_points_homography(match, x1, y1, x2, y2):
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i in range(len(match)):
        if match[i] != -1:
            X1.append(x1[i])
            Y1.append(y1[i])
            X2.append(x2[int(match[i])])
            Y2.append(y2[int(match[i])])
    X1 = np.asarray(X1, dtype=np.int_)
    Y1 = np.asarray(Y1, dtype=np.int_)
    X2 = np.asarray(X2, dtype=np.int_)
    Y2 = np.asarray(Y2, dtype=np.int_)
    
    return X1, Y1, X2, Y2
