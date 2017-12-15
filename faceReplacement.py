import cv2
import PyNet as net
from proj4AdetectFace import detectFace
from scipy.spatial import Delaunay
from proj4AgetFeatures import getFeatures
from scipy.spatial import ConvexHull

from proj3Chelpers import rgb2gray
from proj4AgetFeatures import getFeatures
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from facialLandmark import facialLandmark
#from getConvexHull import getConvexHull

# img1 and img2 are nxn numpy matricies
# noinspection PyInterpreter
def faceReplacement(img1, img2):

    #do face detection and get bounding boxes
    #returns numFacesx4x2 where 4 is four corners and 3rd dimension is xy
    bboxImg1 = detectFace(img1)
    bboxImg2 = detectFace(img2)

    #get facial landmarks from the PyNet CNN
    #returns nx2 locations of the features
    # myModel = net.Model.load_model("Project4BCNN/500.pickle")
    # landmarksImg1 = myModel.forward(img1)
    # landmarksImg2 = myModel.forward(img2)

    #get facial landmarks from dlibs
    #nx2 each
    landmarksImg1 = facialLandmark(img1)
    landmarksImg2 = facialLandmark(img2)
    
    # #getRGB image
    # img1_gray = rgb2gray(img1)
    # img2_gray = rgb2gray(img2)
    #
    # #get the facial features for the image
    # #returns 250xnumFaces points for x locations and y locations
    # xFeatures1, yFeatures1 = getFeatures(img1_gray, bboxImg1)
    # features1 = np.hstack((xFeatures1,yFeatures1))
    # xFeatures2, yFeatures2 = getFeatures(img2_gray, bboxImg2)
    # features2 = np.hstack((xFeatures2,yFeatures2))
    #
    # #get the two convex hulls for the images
    # convexHull1 = cv2.convexHull(features1, returnPoints=True)
    # convexHull2 = cv2.convexHull(features2, returnPoints=True)

    #get the triangulations
    tri1 = Delaunay(landmarksImg1)
    tri2 = Delaunay(landmarksImg2)

    #get the convex hulls
    hull1 = ConvexHull(landmarksImg1)
    hull2 = ConvexHull(landmarksImg2)

    #plot the image
    #plt.imshow(img1)
    #plot the triangulation
    #plt.triplot(landmarksImg1[:, 0], landmarksImg1[:, 1], tri1.simplices.copy())
    #plot the hull
    #plt.plot(landmarksImg1[hull1.vertices, 0], landmarksImg1[hull1.vertices, 1], 'r--', lw=2)
    #plt.show()

    #append the facial landmarks and the convexhulls together
    img1Features = np.vstack((landmarksImg1,hull1.verticies))
    img2Features = np.vstack((landmarksImg2,hull2.verticies))

    #create delanuy triangulation
    tri1 = Delaunay(img1Features)
    tri2 = Delaunay(img2Features)

    #do affine warp of the triangles
    Delaunay.transform()

    #blending if want to
    #we can call this function
    #output = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
