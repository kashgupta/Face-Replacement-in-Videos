'''
  File name: seamlessCloningPoisson.py
  Author: Rajiv POC
  Date created: OCt 15 2017
'''

import numpy as np

from getIndexes import getIndexes
from getCoefficientMatrix import getCoefficientMatrix
from getSolutionVect import getSolutionVect
from reconstructImg import reconstructImg


def seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY):
    targetH, targetW, targetD = targetImg.shape
    indexes = getIndexes(mask, targetH, targetW, offsetX, offsetY)
    coeff_A = getCoefficientMatrix(indexes)
    
    # SolutionVectorb
    solutionVectorb = getSolutionVect(indexes, sourceImg, targetImg, offsetX, offsetY)
    
    #Red
    solVectorb_red = solutionVectorb[:,0]
    red = np.linalg.solve(coeff_A, solVectorb_red)
    red = np.clip(red, 0, 255)
    
    #Green
    solVectorb_green = solutionVectorb[:,1]
    green = np.linalg.solve(coeff_A, solVectorb_green)
    green = np.clip(green, 0, 255)
    
    #Blue
    solVectorb_blue = solutionVectorb[:,2]
    blue = np.linalg.solve(coeff_A, solVectorb_blue)
    blue = np.clip(blue, 0, 255)
    
    resultImg = reconstructImg(indexes, red, green, blue, targetImg)
    
    return resultImg, solutionVectorb