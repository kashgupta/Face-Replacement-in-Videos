'''
  File name: findDerivatives.py
  Author: Rajiv Patel-O'Connor
  Date created: Sept 25, 2017
'''

import numpy as np
from utils import GaussianPDF_2D
from scipy import signal


'''
  File clarification:
    Compute gradient information of the input grayscale image
    - Input I_gray: H x W matrix as image
    - Output Mag: H x W matrix represents the magnitude of derivatives
    - Output Magx: H x W matrix represents the magnitude of derivatives along x-axis
    - Output Magy: H x W matrix represents the magnitude of derivatives along y-axis
    - Output Ori: H x W matrix represents the orientation of derivatives
'''


def findDerivatives(I_gray):
    # create Gaussian filter (11x11 is standard in image processing)
    # sigma = 1.4 taken from Wikipedia (https://en.wikipedia.org/wiki/Canny_edge_detector#Gaussian_filter)
    gaussian_2d = GaussianPDF_2D(1, 1.4, 11, 11)

    # calculate gaussian_2d derivative
    # side note: this can also supposedly be calculated using sobel operators?
    [dy, dx] = np.gradient(gaussian_2d)

    # smooth grayscale image using gaussian filter
    smoothed_image = signal.convolve(I_gray, gaussian_2d, mode="same")

    # get derivatives of I_gray in horizontal and vertical directions by convolving with derivative of gaussian_2d
    Magx = signal.convolve(smoothed_image, dx, mode="same")
    Magy = signal.convolve(smoothed_image, dy, mode="same")

    # calculate edge gradient and orientation
    # np.arctan2() goes from -pi to pi
    Mag = np.sqrt(np.square(Magx) + np.square(Magy))
    Ori = np.arctan2(Magy, Magx)

    return Mag, Magx, Magy, Ori


