'''
  File name: nonMaxSup.py
  Author: Rajiv Patel-O'Connor
  Date created: Sept 25, 2017
'''

import numpy as np
from interp2 import interp2

'''
  File clarification:
    Find local maximum edge pixel using NMS along the line of the gradient
    - Input Mag: H x W matrix represents the magnitude of derivatives
    - Input Ori: H x W matrix represents the orientation of derivatives
    - Output M: H x W binary matrix represents the edge map after non-maximum suppression
'''


def nonMaxSup(Mag, Ori):

    pi = 3.1415926535897932385;

    # create meshgrid for x and y
    [height, width] = Mag.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    [X, Y] = np.meshgrid(x, y)

    # find subpixel coordinates for interpolation based on orientation
    qx1 = np.cos(Ori) + X
    qy1 = np.sin(Ori) + Y
    qx2 = np.cos(Ori + pi) + X
    qy2 = np.sin(Ori + pi) + Y

    # interpolation (MUCH BETTER THAN DISCRETIZING AND THEN INTERPOLATING)
    p_mag = interp2(X, Y, Mag, qx1, qy1)
    r_mag = interp2(X, Y, Mag, qx2, qy2)

    # determine whether gradient magnitude is larger than subpixel interpolated gradient magnitudes
    diff_p_mag = Mag - p_mag
    diff_r_mag = Mag - r_mag
    local_max = np.asarray(np.where((diff_p_mag > 0) & (diff_r_mag > 0)))
    local_max_as_list = local_max.tolist()

    # return output as binary
    M = np.zeros((height, width), dtype=np.bool)
    M[local_max_as_list] = True

    return M

