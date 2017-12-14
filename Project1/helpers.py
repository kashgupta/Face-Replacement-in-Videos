'''
  File name: helpers.py
  Author: Rajiv Patel-O'Connor
  Date created: Sept 25, 2017
'''

import numpy as np
import matplotlib.pyplot as plt
from cannyEdge import cannyEdge

'''
  File clarification:
    Helpers file that contributes the project
    You can design any helper function in this file to improve algorithm
'''


def color_map(image):

    # First get the edges
    E = cannyEdge(image)
    [x, y] = np.where(E == 1)

    # Add color
    color = np.zeros(image.shape, dtype=np.int8)
    color[x, y, :] = image[x, y, :]

    # plt.imshow(color)
    # plt.show()

    return color
