#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:37:06 2017

@author: rajivpatel-oconnor
"""
import matplotlib.pyplot as plt
import numpy as np
from myvideomosaic import myvideomosaic

ids = [1,2,3, 4, 11,12,13,14,15,16,17,18,19, 21, 22, 23, 24, 25, 31,41,42,43,44,45,46,47,48,49,50,51,52,61,62,63, 64] 
allLoaded = []
for i in range(len(ids)):
    img =plt.imread('./roboOut/finalFusedCol' + str(ids[i]) + '.jpg')
    allLoaded.append(img[:, :, 0:3])
img_mosaic = np.asarray(allLoaded)
myvideomosaic(img_mosaic)
