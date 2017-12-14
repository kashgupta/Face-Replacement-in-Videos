'''
    File name: getCoefficientMatrix.py
    Author: Rajiv Patel-O'Connor
    Date created: Oct 16, 2017
    '''

import numpy as np


def getCoefficientMatrix(indexes):
    
    max_index = np.amax(indexes)
    coeff_A = np.zeros((max_index, max_index), dtype=np.int32)
    rows, columns = np.nonzero(indexes)
    
    for i in range(0, max_index):
        coeff_A[i][i] = 4
        
        #Check above
        if (rows[i] > 0) and (indexes[rows[i] - 1][columns[i]] != 0):
            coeff_A[i][indexes[rows[i] - 1][columns[i]]-1] = -1
    
        #Check below
        if (rows[i] < max_index - 1) and (indexes[rows[i] + 1][columns[i]] != 0):
            coeff_A[i][indexes[rows[i] + 1][columns[i]]-1] = -1
        
        #Check right
        if (columns[i] < max_index - 1) and (indexes[rows[i]][columns[i] + 1] != 0):
            coeff_A[i][indexes[rows[i]][columns[i]+ 1]-1] = -1
        
        #Check left
        if (columns[i] > 0) and (indexes[rows[i]][columns[i] - 1] != 0):
            coeff_A[i][indexes[rows[i]][columns[i] - 1]-1] = -1
    return coeff_A
