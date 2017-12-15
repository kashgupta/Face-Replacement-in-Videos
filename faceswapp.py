#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:48:41 2017

@author: rajivpatel-oconnor
"""

import sys
import numpy as np
import cv2
import dlib
from facialLandmark import facialLandmark
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

# # Read points from text file
# def readPoints(path) :
#     # Create an array of points.
#     points = [];
#
#     # Read points
#     with open(path) as file :
#         for line in file :
#             x, y = line.split()
#             points.append((int(x), int(y)))
#
#
#     return points


#
# # Check if a point is inside a rectangle
# def rectContains(rect, point) :
#     if point[0] < rect[0] :
#         return False
#     elif point[1] < rect[1] :
#         return False
#     elif point[0] > rect[0] + rect[2] :
#         return False
#     elif point[1] > rect[1] + rect[3] :
#         return False
#     return True


# #calculate delanauy triangle
# def calculateDelaunayTriangles(rect, points):
#     #create subdiv
#     subdiv = cv2.Subdiv2D(rect);
#
#     # Insert points into subdiv
#     for p in points:
#         subdiv.insert(p)
#
#     triangleList = subdiv.getTriangleList();
#
#     delaunayTri = []
#
#     pt = []
#
#     count= 0
#
#     for t in triangleList:
#         pt.append((t[0], t[1]))
#         pt.append((t[2], t[3]))
#         pt.append((t[4], t[5]))
#
#         pt1 = (t[0], t[1])
#         pt2 = (t[2], t[3])
#         pt3 = (t[4], t[5])
#
#         if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
#             count = count + 1
#             ind = []
#             for j in xrange(0, 3):
#                 for k in xrange(0, len(points)):
#                     if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
#                         ind.append(k)
#             if len(ind) == 3:
#                 delaunayTri.append((ind[0], ind[1], ind[2]))
#
#         pt = []
#
#
#     return delaunayTri
#

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
    

if __name__ == '__main__' :

    # detector = dlib.get_frontal_face_detector()  # Face detector
    # predictor = dlib.shape_predictor(
    #     "shape_predictor_68_face_landmarks.dat")  # Landmark identifier. Set the filename to whatever you named the downloaded file

    #
    # # Get the face coordinates in Image1
    # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # # #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # # #clahe_image = clahe.apply(gray)
    # points1 = []
    #
    # detections = detector(gray, 1) #Detect the faces in the image
    # for k,d in enumerate(detections): #For each detected face
    #     shape = predictor(gray, d) #Get coordinates
    #     for i in range(1,68): #There are 68 landmark points on each face
    #         points1.append((shape.part(i).x, shape.part(i).y))
    #
    # # Get the face coordinates in Image2
    # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # clahe_image = clahe.apply(gray)
    # points2 = []
    #
    # detections = detector(clahe_image, 1) #Detect the faces in the image
    # for k,d in enumerate(detections): #For each detected face
    #     shape = predictor(clahe_image, d) #Get coordinates
    #     for i in range(1,68): #There are 68 landmark points on each face
    #         points2.append((shape.part(i).x, shape.part(i).y))
    #
    # print points1

    # Find convex hull
    #hull1 = []
    #hull2 = []
    # hull1 = np.zeros(indices.shape[0],2)
    # for i in xrange(0, len(indices)):
    #
    #     hull1 = .append(points1[int(indices[i])])
    #     hull2.append(points2[int(indices[i])])


    # # Find delanauy traingulation for convex hull points
    # sizeImg2 = img2.shape
    # rect = (0, 0, sizeImg2[1], sizeImg2[0])
    # dt = calculateDelaunayTriangles(rect, hull2)

    # Calculate Mask
    # hull8U = []
    # #hull8U = np.zeros(hull2.shape)
    # for i in range(0, len(hull2)):
    #     hull8U.append((hull2[i][0], hull2[i][1]))


    # Read images
    #filename1 = './data/easy/MarquesBrownlee0.jpg'
    #filename2 = './data/easy/TheMartian0.jpg'
    file1= 'ted_cruz.jpg'
    file2= 'donald_trump.jpg'
    img1 = cv2.imread(file1);
    img2 = cv2.imread(file2);
    img1Warped = np.copy(img2);

    #find the landmarks
    landmarks1 = facialLandmark(img1)
    landmarks2 = facialLandmark(img2)

    #get the convex hull
    indices = cv2.convexHull(np.array(landmarks2), returnPoints = False)
    hull1 = landmarks1[indices[:,0]]
    hull2 = landmarks2[indices[:,0]]
    #convert hulls to tuple list
    hull1 = tuple(map(tuple, hull1))
    hull2 = tuple(map(tuple, hull2))

    #create delanuy triangulation
    tri = Delaunay(hull2).simplices
    #conver to tuple list
    tri = tuple(map(tuple, tri))

    #for each of the triangles, do the affine warp
    numTriangles = len(tri)
    for i in range(0, numTriangles):
        triangle = tri[i]
        pts1 = np.zeros((3, 2), dtype=np.float32)
        pts2 = np.zeros((3, 2), dtype=np.float32)

        # calculate transform for each triangle in both directions
        pts1[0] = hull1[triangle[0]]
        pts1[1] = hull1[triangle[1]]
        pts1[2] = hull1[triangle[2]]

        pts2[0] = hull2[triangle[0]]
        pts2[1] = hull2[triangle[1]]
        pts2[2] = hull2[triangle[2]]

        warpTriangle(img1, img1Warped, pts1, pts2)

    #do blending
    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))

    #find the center of the face to do the blending
    r = cv2.boundingRect(np.float32([hull2]))
    faceCenter = (r[0]+int(r[2]/2), r[1]+int(r[3]/2))

    finalImage = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, faceCenter, cv2.NORMAL_CLONE)
    
    cv2.imshow('image', finalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()