'''
  File name: anms.py
  Author: Rajiv Patel-O'Connor
  Date created: Nov 7, 2017
'''
import numpy as np
from Project3C.helpers import distance


def anms(cimg, max_pts):
    nonzero = np.nonzero(cimg)
    nz_rows = nonzero[0]
    nz_cols = nonzero[1]
    num_features = len(nonzero[0])
    threshold = 1
    store = np.zeros((num_features, 3))
    
    #implement a data structure to speed this up
    
    for i in range(num_features):
        minRadius = 100000
        comparativeIntensity = cimg[nz_rows[i]][nz_cols[i]]
        for j in range(num_features):
            if i == j:
                continue
            if cimg[nz_rows[j]][nz_cols[j]] >= comparativeIntensity*threshold:
                currRadius = distance(nz_rows[j], nz_cols[j], nz_rows[i], nz_cols[i])
                if currRadius < minRadius:
                    minRadius = currRadius
        store[i][0] = nz_rows[i]
        store[i][1] = nz_cols[i]
        store[i][2] = minRadius
        
    store = store[store[:, 2].argsort()][::-1]
    
    if max_pts > num_features:
        y = store[:, 0]
        x = store[:, 1]
        rmax = np.min(store[:,2])
        return x, y, rmax
    
    y = store[:, 0][:max_pts].astype(int)
    x = store[:, 1][:max_pts].astype(int)
    rmax  = store[:,2][max_pts + 1]
          
    return x, y, rmax

    