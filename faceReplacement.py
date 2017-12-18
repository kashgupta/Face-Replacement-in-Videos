import cv2
from scipy.spatial import Delaunay
import numpy as np
from facialLandmark import facialLandmark
from warpTriangle import warpTriangle

def faceReplacement(img1, img2, oldFacialLandmarks1, oldFacialLandmarks2):
    img1Warped = np.copy(img2);

    #find the landmarks for img1
    try:
        landmarks1 = facialLandmark(img1)
    except UnboundLocalError:
        landmarks1 = oldFacialLandmarks1

    #find the landmarks for img2
    try:
        landmarks2 = facialLandmark(img2)
    except UnboundLocalError:
        landmarks2 = oldFacialLandmarks2

    #get the convex hull
    indices = cv2.convexHull(np.array(landmarks2), returnPoints = False)
    hull1 = landmarks1[indices[:,0]]
    hull2 = landmarks2[indices[:,0]]

    #create delanuy triangulation
    tri = Delaunay(hull2).simplices

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

    #get the mask of only the hull for blending
    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))

    #find the center of the face to do the blending
    r = cv2.boundingRect(np.float32([hull2]))
    faceCenter = (r[0]+int(r[2]/2), r[1]+int(r[3]/2))

    #do the blending
    finalImage = cv2.seamlessClone(np.uint8(img1Warped), np.uint8(img2), mask, faceCenter, cv2.NORMAL_CLONE)

    #conert the image to a RGB image from BGR image that cv2 returns
    return cv2.cvtColor(np.uint8(finalImage), cv2.COLOR_BGR2RGB), landmarks1, landmarks2