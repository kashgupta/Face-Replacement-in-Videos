'''
  File name: reconstructImg.py
  Author: Rajiv Patel-O'Connor
  Date created:
'''
import numpy as np

def reconstructImg(indexes, red, green, blue, targetImg):
    max_index = np.amax(indexes)
    rows, columns = np.nonzero(indexes)
    
    for i in range(0, max_index):
        targetImg[rows[i]][columns[i]] = [red[i], green[i], blue[i]]
    
    return targetImg