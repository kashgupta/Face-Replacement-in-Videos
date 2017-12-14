import cv2
import PyNet as net
from detectFace import detectFace
from scipy.spatial import Delaunay
from Project4A.getFeatures import getFeatures
import numpy as np
#from getConvexHull import getConvexHull

# img1 and img2 are nxn numpy matricies
# noinspection PyInterpreter
def faceReplacement(img1, img2):

    #do face detection and get bounding boxes
    #returns numFacesx4x2 where 4 is four corners and 3rd dimension is xy
    bboxImg1 = detectFace(img1)
    bboxImg2 = detectFace(img2)

    #get facial landmarks
    #returns nx2 locations of the features
    myModel = net.Model.load_model("Project4BCNN/500.pickle")
    landmarksImg1 = myModel.forward(img1)
    landmarksImg2 = myModel.forward(img2)

    #get the facial features for the image
    #returns 250xnumFaces points for x locations and y locations
    xFeatures1, yFeatures1 = getFeatures(img1, bboxImg1)
    features1 = np.hstack((xFeatures1,yFeatures1))
    xFeatures2, yFeatures2 = getFeatures(img2, bboxImg2)
    features2 = np.hstack((xFeatures2,yFeatures2))

    #get the two convex hulls for the images
    convexHull1 = cv2.convexHull(features1, returnPoints=False)
    convexHull2 = cv2.convexHull(features2, returnPoints=False)

    #append the facial landmarks and the convexhulls together
    img1Features = np.vstack((landmarksImg1,convexHull1))
    img2Features = np.vstack((landmarksImg2,convexHull2))

    #create delanuy triangulation
    tri1 = Delaunay(img1Features)
    tri2 = Delaunay(img2Features)

    #do affine warp of the triangles
    Delaunay.transform()

    #blending if want to
    #we can call this function
    #output = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
