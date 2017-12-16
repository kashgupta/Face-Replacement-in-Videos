import cv2
import numpy as np


def videoToNumpy(filepath):
    vidcap = cv2.VideoCapture(filepath)
    success,image = vidcap.read()
    success = True
    listOfFrames = []
    while success:
        success,image = vidcap.read()
        listOfFrames.append(image) # add to list
    return listOfFrames