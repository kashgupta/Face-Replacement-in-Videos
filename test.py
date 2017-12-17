from faceReplacementForVideo import faceReplacementForVideo
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from proj4AdetectFace import detectFace
from scipy.spatial import ConvexHull
from facialLandmark import facialLandmark
from scipy.spatial import Delaunay
import cv2
import numpy as np
import skvideo.io

def main():

    video1 = "./Data/Easy/TheMartian.mp4"
    video2 = "./Data/Easy/MarquesBrownlee.mp4"
    #file1= 'ted_cruz.jpg'
    #file2= 'donald_trump.jpg'

    faceReplacementForVideo(video1, video2)
    #skvideo.io.vwrite("outputvideo.mp4", swappedVid21)
    #imageio.mimwrite('swap21.mp4', swappedVid21, fps=30)

if __name__ ==  "__main__":
    main()