'''
  File name: helpers.py
  Author: Rajiv Patel-O'Connor
  Date created: 10-18-2017
'''

import numpy as np


def calculateAFromIdx(index, simplices, pts):

    # Get vertices
    vtx1 = simplices[pts[index][0]]
    vtx2 = simplices[pts[index][1]]
    vtx3 = simplices[pts[index][2]]
    
    A = [
         [vtx1[0], vtx2[0], vtx3[0]],
         [vtx1[1], vtx2[1], vtx3[1]],
         [1, 1, 1]
         ]
    return np.asarray(A)


def get_rgb(x, y, im):
    return im[int(round(y)), int(round(x))]
