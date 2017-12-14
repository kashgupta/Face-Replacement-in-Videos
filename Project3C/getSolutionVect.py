'''
  File name: getSolutionVect.py
  Author: Rajiv Patel-O'Connor
  Date created: Oct 16, 2017
'''

import numpy as np
import scipy.signal as signal

def getSolutionVect(indexes, source, target, offsetX, offsetY):
    index_max = np.amax(indexes)
    SolVectorb = np.empty([index_max, 3], dtype=np.int32)
    [h, w, d] = target.shape
    rows, columns = np.nonzero(indexes)
    kernel = np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])    
    
    for j in range(0, d):
        source_curr = source[:,:,j]
        target_curr = target[:,:, j]
        source_laplacian = signal.convolve2d(source_curr, kernel, mode='same')
        for i in range(0, index_max):
            
            solution_value = 0;
            #Check above
            if (rows[i] == 0) or (indexes[rows[i] - 1][columns[i]] == 0):
                if  rows[i] - 1 >= 0:
                    solution_value += target_curr[ rows[i] - 1][ columns[i] ]
            
            #Check below
            if (rows[i] == index_max - 1) or (indexes[rows[i] + 1][columns[i]] == 0):
                if  rows[i] + 1 <= h - 1:
                    solution_value += target_curr[ rows[i] + 1][ columns[i]]
        
            #Check right
            if (columns[i] == index_max - 1) or (indexes[rows[i]][columns[i] + 1] == 0):
                if  columns[i] + 1 <= w - 1:
                    solution_value += target_curr[ rows[i]][ columns[i] + 1]
                    
            #Check left
            if (columns[i] == 0) or (indexes[rows[i]][columns[i] - 1] == 0):
                if columns[i]  - 1 >= 0:
                    solution_value += target_curr[ rows[i]][ columns[i] - 1]
            SolVectorb[i][j] = solution_value + source_laplacian[rows[i]-offsetY, columns[i]-offsetX]
        
    return SolVectorb
