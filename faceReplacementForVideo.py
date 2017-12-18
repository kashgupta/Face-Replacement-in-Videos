import numpy as np
from faceReplacement import faceReplacement
from helpers4C import videoToNumpy
import skvideo.io

def faceReplacementForVideo(video1, video2):

    frames1 = videoToNumpy(video1)
    frames2 = videoToNumpy(video2)
    landmarks1 = [];
    landmarks2 = [];

    print len(frames1)
    print len(frames2)
    #Allocate Memory for swapped vid 1
    [h1, w1, _] = frames1[0].shape
    swappedVid1 = np.zeros((len(frames1) - 1, h1, w1, 3), dtype=np.uint8)

    #Generate frames for swapped vid 1
    for i in range(len(frames1) - 1):
        if i < len(frames2) - 1: # handle when vid1 is longer than vid1
            img1 = frames2[i]
        else:
            img1 = frames2[len(frames2) - 2]
        print(i)
        swappedVid1[i,:,:,:], landmarks1, landmarks2 = faceReplacement(img1, frames1[i], landmarks1, landmarks2)


    # #Allocate Memory for swapped vid 2
    # landmarks1 = []
    # landmarks2 = []
    # [h2, w2, _] = frames2[0].shape
    # swappedVid2 = np.zeros((len(frames2) - 1, h2, w2, 3), dtype=np.uint8)
    #
    # #Generate frames for swapped vid 2
    # for j in range(len(frames2) - 1):
    #     if j < len(frames1) - 1: #handle when vid2 is longer than vid 1
    #         img1 = frames1[j]
    #     else:
    #         img1 = frames1[len(frames1) - 2]
    #     print j
    #     swappedVid2[j,:,:,:], landmarks1, landmarks2 = faceReplacement(img1, frames2[j], landmarks1, landmarks2)



    #save as a video
    skvideo.io.vwrite("marquesbrownlee_on_themartian.mp4", swappedVid1)
    #skvideo.io.vwrite("themartian_on_marquesbrownlee.mp4", swappedVid2)

