from faceReplacement import faceReplacement
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from proj4AdetectFace import detectFace
from scipy.spatial import ConvexHull
from facialLandmark import facialLandmark
from scipy.spatial import Delaunay
import cv2
import numpy as np

def main():

    img1File = "Data/Easy/TheMartian0.jpg"
    img2File = "Data/Easy/MarquesBrownlee0.jpg"

    img1=mpimg.imread(img1File)
    img2=mpimg.imread(img2File)

    img1Warped = np.copy(img2);

    [newImg1, newImg2] = faceReplacement(img1, img2)

    # bboxImg1 = detectFace(img1)
    # bboxImg2 = detectFace(img2)
    # plt.imshow(img1)
    #
    # pts = facialLandmark(img1)
    #
    # tri1 = Delaunay(pts)

    #print tri1.simplices[0,:]

    #plt.triplot(pts[:, 0], pts[:, 1], tri1.simplices.copy())

    #hull = ConvexHull(pts)

    #plt.plot(pts[hull.vertices, 0], pts[hull.vertices, 1], 'r--', lw=2)
    #plt.show()

if __name__ ==  "__main__":
    main()