#from faceReplacement import faceReplacement
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from proj4AdetectFace import detectFace
from facialLandmark import facialLandmark
import cv2
import numpy as np

def main():

    img1File = "Data/Easy/TheMartian0.jpg"
    img2File = "Data/Easy/MarquesBrownlee0.jpg"

    img1=mpimg.imread(img1File)
    img2=mpimg.imread(img2File)

    #[newImg1, newImg2] = faceReplacement(img1, img2)

    bboxImg1 = detectFace(img1)
    bboxImg2 = detectFace(img2)

    pts = facialLandmark(img1)


if __name__ ==  "__main__":
    main()