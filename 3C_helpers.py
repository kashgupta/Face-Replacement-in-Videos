#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:35:15 2017

@author: rajivpatel-oconnor
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from seamlessCloningPoisson import seamlessCloningPoisson
from time import gmtime, strftime


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def apply_homography(H, x, y):
    matrix = np.stack((x, y, np.ones(len(x))))
    applied_homography = H.dot(matrix)
    X = np.divide(applied_homography[0], applied_homography[2])
    Y = np.divide(applied_homography[1], applied_homography[2])
    return X, Y

def count_inliers(new_x1, new_y1, x2, y2, threshold):
    inlier_binary = np.zeros((len(new_x1)), dtype=np.bool_)
    count = 0
    for i in range(len(new_x1)):
        if distance(new_y1[i], new_x1[i], y2[i], x2[i]) <= threshold:
            count += 1
            inlier_binary[i] = 1
    return count, inlier_binary

def distance(y1, x1, y2, x2):
    return math.sqrt((x2-x1)**2 + (y2- y1)**2)

def overlay_points(img, x, y, name):
    plt.figure()
    implot = plt.imshow(img)
    plt.scatter(x, y,color='red',marker='o', s=1)
    plt.savefig('postanms' + name)
    plt.close("all")
    
def obtain_points_homography(match, x1, y1, x2, y2):
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    for i in range(len(match)):
        if match[i] != -1:
            X1.append(x1[i])
            Y1.append(y1[i])
            X2.append(x2[int(match[i])])
            Y2.append(y2[int(match[i])])
    X1 = np.asarray(X1, dtype=np.int_)
    Y1 = np.asarray(Y1, dtype=np.int_)
    X2 = np.asarray(X2, dtype=np.int_)
    Y2 = np.asarray(Y2, dtype=np.int_)
    
    return X1, Y1, X2, Y2

def middle_left_fusion(leftIm, middleIm, leftIm_color, middleIm_color, H, inlier_ind, X1, Y1, X2, Y2):
    #homography from one to two,
    #in this case homography from 2>3
    #since we're actually changing 3>2 we take inverse!
    H=np.linalg.inv(H)
    
    #warp non-center, find edges of this
    [h, w] = leftIm.shape
    x = np.linspace(0, w-1, 2)
    y = np.linspace(0, h-1, 2)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    
    #rearrange so polygon order
    xv=xv.reshape((4,))
    xv[3], xv[2] = xv[2], xv[3]
    
    yv=yv.reshape((4,))
    yv[3], yv[2] = yv[2], yv[3]
    
    #map to warped (2>3)
    [X, Y] = apply_homography(H, xv, yv)
    
    #find offset of warped image
    Yoff = np.amin(Y)
    Xoff = np.amin(X)
    
    #move to domain greater than 0 (D>=0)
    nY = Y-Yoff
    nX = X-Xoff
    
    #find max indices with (D>=0)
    nw = int(np.ceil(np.amax(nX)))
    nh = int(np.ceil(np.amax(nY)))
    
    'find indices within vertices'
    #list of vertices, input to polygon function
    verts=[]
    for i in range(4):
        verts.append((nX[i],nY[i]))
    
    
    x, y = np.meshgrid(np.arange(nw), np.arange(nh)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    
    p = Path(verts) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(nh, nw)
    
    plt.figure()
    plt.imshow(mask)
    
    maskY,maskX=np.nonzero(mask)
    maskYoff=maskY+Yoff
    maskYoff=maskYoff.astype(int)
    
    #maskXoff is indices in warped image
    maskXoff=maskX+Xoff
    maskXoff=maskXoff.astype(int)
    
    #backmap indices to find what points should be used
    Hinv=np.linalg.inv(H)
    [srcX, srcY] = apply_homography(Hinv, maskXoff, maskYoff)
    srcXint=srcX.astype(int)
    srcYint=srcY.astype(int)
    
    im = np.zeros((nh, nw),dtype='uint8')
    im_color = np.zeros((nh, nw, 3),dtype='uint8')
    
    #here should use srcX and interpolate!!
    im[maskY,maskX]=leftIm[srcYint,srcXint]
    im_color[maskY,maskX]=leftIm_color[srcYint,srcXint]

    '????'
    #consider inlier coordinated
    X1, Y1, X2, Y2 = X1[inlier_ind], Y1[inlier_ind], X2[inlier_ind], Y2[inlier_ind]
    #warp the right  inlier coords
    ''
    [nX2, nY2] = apply_homography(H, X2, Y2)
    
    XPtsOnNewImR=nX2-Xoff
    YPtsOnNewImR=nY2-Yoff
    #find offset in fused image needed to line up the inliers
    Xoff2=X1-XPtsOnNewImR
    Xoff2=int(np.mean(Xoff2))
    Yoff2=Y1-YPtsOnNewImR
    Yoff2=int(np.mean(Yoff2))
    
    yToOffset=np.argmin((0,Yoff2))
    xToOffset=np.argmin((0,Xoff2))
    [h, w] = middleIm.shape
    
    hmx=max(h,nh+Yoff2)
    hmn=min(0,Yoff2) #YfusedOffset
    wmx=max(w,nw+Xoff2)
    wmn=min(0,Xoff2) #XfusedOffset
    
    hfused=int(hmx-hmn)
    wfused=int(wmx-wmn)
    fusedIm=np.zeros((hfused,wfused),dtype=np.uint8)
    fusedImColor=np.zeros((hfused,wfused,3),dtype=np.uint8)
    
    [h, w] = middleIm.shape
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    xv=xv.astype(int)
    yv=yv.astype(int)
    xpostoffset = xv if (xToOffset==0) else  xv-Xoff2
    ypostoffset = yv if (yToOffset==0) else yv-Yoff2
    fusedIm[ypostoffset.astype(int),xpostoffset.astype(int)]=middleIm[yv,xv]
    fusedImColor[ypostoffset.astype(int),xpostoffset.astype(int)]=middleIm_color[yv,xv]
    plt.figure()
    plt.imshow(fusedIm)
    
    #save for finding perimeter later
    oldypostoffset=ypostoffset
    oldxpostoffset=xpostoffset
    
    ypostoffset=ypostoffset.reshape(h*w,)
    xpostoffset=xpostoffset.reshape(h*w,)
    
    oldYX = np.stack((ypostoffset.astype(int), xpostoffset.astype(int)))
    oldYX = oldYX.T
    #oldS is shape of first inserted image in new space
    oldS=set()
    for oldTup in oldYX:
        oldS.add(tuple(oldTup))
    
    xpostoffset = maskX if (xToOffset==1) else maskX+Xoff2
    ypostoffset = maskY if (yToOffset==1) else maskY+Yoff2
    
    newYX = np.stack((ypostoffset.astype(int), xpostoffset.astype(int)))
    newYX = newYX.T
    
    newS=set()
    for newTup in newYX:
        newS.add(tuple(newTup))
    intersection = np.array([tup for tup in oldS & newS])
    
    interSet = set()
    for tup in intersection:
        interSet.add(tuple(tup))
    
    
    
    diff = newS.difference(interSet)
    
    diffAr=np.array(list(diff))

    
    
    maskX = diffAr[:,1] if (xToOffset==1) else diffAr[:,1]-Xoff2
    maskY = diffAr[:,0] if (yToOffset==1) else diffAr[:,0]-Yoff2
    
    
    fusedIm[diffAr[:,0],diffAr[:,1]]=im[maskY,maskX]
    fusedImColor[diffAr[:,0],diffAr[:,1]]=im_color[maskY,maskX]
    plt.figure()
    plt.imshow(fusedIm)
    tm = strftime("%H_%M_%S", gmtime())
    
    plt.figure()
    plt.imshow(fusedImColor)
    plt.imsave('color'+tm+'col.jpg',fusedImColor)

    
    
    
    
    
    
    
    
    
    
    #find perimeter of middle image
    perimS=set()
    for i in range(h):
        t = tuple([oldypostoffset[i,0], oldxpostoffset[i,0]])
        perimS.add(t)
        t = tuple([oldypostoffset[i,w-1], oldxpostoffset[i,w-1]])
        perimS.add(t)
    for j in range(w):
        t = tuple([oldypostoffset[0,j], oldxpostoffset[0,j]])
        perimS.add(t)
        t = tuple([oldypostoffset[h-1,j], oldxpostoffset[h-1,j]])
        perimS.add(t)
    
    #intersection of intersection and perimeter of middle image
    perimeter = np.array([tup for tup in interSet & perimS])
    
    mask = np.zeros((hfused,wfused),dtype=np.bool_)
    tmp = perimeter
    mask[tmp[:,0],tmp[:,1]]=True
    plt.imshow(mask)
    
    
    [ph,pw]=perimeter.shape
    
    #expand this perimeter in both directions
    GDBmask = np.zeros((hfused,wfused),dtype=np.bool_)
    fld = 15
    ymi = perimeter[:,0]-fld
    ymi[ymi<0]=0
    
    yma=perimeter[:,0]+fld
    yma[yma>hfused-1]=hfused-1
    
    xmi=perimeter[:,1]-fld
    xmi[xmi<0]=0
    
    xma=perimeter[:,1]+fld
    xma[xma>wfused-1]=wfused-1
    
    for i in range(ph):
        GDBmask[ymi[i]:yma[i],xmi[i]:xma[i]]=True
    [ynz,xnz]=np.nonzero(GDBmask)
    
    
    
    #intersection of oldset and expanded overlap_perim
    maskS = set()
    for i in range(len(ynz)):
        maskS.add(tuple([ynz[i],xnz[i]]))
    toBlend = np.array([tup for tup in maskS & interSet])
    


    mask = np.zeros(im.shape,dtype=np.bool_)

    offsetX = 0 if (xToOffset==1) else Xoff2
    offsetY = 0 if (yToOffset==1) else Yoff2

    mask[toBlend[:,0]-offsetY,toBlend[:,1]-offsetX]=True

    
    
    [resultImg, solutionVectorb] = seamlessCloningPoisson(im_color,fusedImColor, mask, offsetX, offsetY)

    plt.imsave('poissonBlended'+tm+'.jpg',resultImg)
    
    
    return fusedIm, resultImg