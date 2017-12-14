'''
  File name: feat_match.py
  Author:
  Date created:
'''
import numpy as np
import math

def feat_match(descs1, descs2):
    [sf,n1] = descs1.shape
    [sf,n2] = descs2.shape
    match = np.zeros((n1,1))
    # for each point
    for i in range(n1):
        v1 = descs1[:,i]
        # find two nearest in n2
        sizes=np.zeros((n1))
        for j in range(n2):
            v2 = descs2[:,j]
            sizes[j] = SSD(v1, v2)
        srt=np.sort(sizes)
        if (srt[0]/srt[1] < 0.6):
            match[i] = np.where(srt[0]==sizes)
        else:
            match[i] = -1
    return match

def SSD(v1, v2):
    return math.sqrt(np.sum((v1-v2)**2))
