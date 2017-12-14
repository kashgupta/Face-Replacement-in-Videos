'''
  File name: getIndexes.py
  Author:
  Date created:
'''


'''
mask is size of src image
targetH is height of target im
targetW is width of target image
offsetX is offset of src
offsetY is offset of src 

output: for the target image size h',w' return amn array with of aach replacement pixel
'''
import numpy as np

import matplotlib.pyplot as plt

def getIndexes(mask, targetH, targetW, offsetX, offsetY):
# Enter Your Code Here
    [h,w]=mask.shape
    im=mask.astype(int)
    ind=0;
    for y in range(h):
        for x in range(w):
            if (im[y][x]==1):
                ind=ind+1;
                im[y][x]=ind;
    plt.imshow(im)
    
    
    indexes=np.zeros((targetH,targetW))
    indexes[offsetY:offsetY+h,offsetX:offsetX+w]=im;
    
    print(np.amax(indexes))
    indexes = indexes.astype(dtype=np.int32)
    
    return indexes.astype(dtype=np.int32)


    '''
    fullMask=np.zeros((h,w),dtype=bool)
    
    
    #img[bbox[2],bbox[4]][bbox[1],bbox[3]]=mask
    '''