'''
  File name: ransac_est_homography.py
  Author: Rajiv Patel-O'Connor
  Date created: Nov 7, 2017
'''

import numpy as np
from random import randint, sample
from est_homography import est_homography
from helpers import apply_homography, count_inliers

def ransac_est_homography(X1, Y1, X2, Y2, thresh):
    maxCount = 0
    H = np.zeros((3,3))
    pts_ransac = np.zeros((4, 4), dtype = np.int_)
    inlier_ind = np.zeros((len(X1)), dtype = np.bool_)
    for k in range(100000):
        smpl=sample(range(len(X1)), 4)
        for i in range(4):
            rand_idx = smpl[i]
            rand_idx = randint(0, len(X1) - 1)
            pts_ransac[i][0] = X1[rand_idx]
            pts_ransac[i][1] = Y1[rand_idx]
            pts_ransac[i][2] = X2[rand_idx]
            pts_ransac[i][3] = Y2[rand_idx]
        
        currH = est_homography(pts_ransac[:, 0], pts_ransac[:, 1], pts_ransac[:, 2], pts_ransac[:, 3])
        new_X1, new_Y1 = apply_homography(currH, X1, Y1)
        count, inlier_binary = count_inliers(new_X1, new_Y1, X2, Y2, thresh)
    
        if count > maxCount:
            maxCount = count
            H = currH
            inlier_ind = inlier_binary
            
        #if maxCount > 10:
          #  break
    
    print(maxCount)
    return H, inlier_ind