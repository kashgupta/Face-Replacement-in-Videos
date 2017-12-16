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

    img2File = "Data/Easy/TheMartian0.jpg"
    img1File = "Data/Easy/MarquesBrownlee0.jpg"
    #file1= 'ted_cruz.jpg'
    #file2= 'donald_trump.jpg'

    [newImg1] = faceReplacement(img1File, img2File)

if __name__ ==  "__main__":
    main()