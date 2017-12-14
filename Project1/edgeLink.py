'''
  File name: edgeLink.py
  Author: Rajiv Patel-O'Connor
  Date created: Sept 25, 2017
'''

import numpy as np

'''
  File clarification:
    Use hysteresis to link edges based on high and low magnitude thresholds
    - Input M: H x W logical map after non-max suppression
    - Input Mag: H x W matrix represents the magnitude of gradient
    - Input Ori: H x W matrix represents the orientation of gradient
    - Output E: H x W binary matrix represents the final canny edge detection map
'''


def edgeLink(M, Mag, Ori):
    
    strong_edges_x, strong_edges_y, super_weak_x, super_weak_y = find_strong_and_super_weak(M, Mag)

    # find weak edges and make subsequent meshgrid
    weak_edges = M
    weak_edges[strong_edges_x, strong_edges_y] = 0
    weak_edges[super_weak_x, super_weak_y] = 0
    connected_edge_meshgrid = generate_padded_meshgrid(weak_edges)

    # Make list to utilize list properties
    strong_edges_x = strong_edges_x.tolist()
    strong_edges_y = strong_edges_y.tolist()

    # define relevant variables for checking connectivity of weak edges to strong edges
    inspected = 0
    strong_edges_num = len(strong_edges_x)

    # find overall connectivity
    while inspected < strong_edges_num:
        x = strong_edges_x[inspected]
        y = strong_edges_y[inspected]
        connected_strong_edges = np.where(connected_edge_meshgrid[:, x, y] == 1)
        connected_strong_edges = connected_strong_edges[0]
        if np.size(connected_strong_edges) > 0:
            for i in connected_strong_edges:
                [new_x, new_y] = meshgrid_shift(x, y, i, strong_edges_x, strong_edges_y, strong_edges_num, M)
                connected_edge_meshgrid = update_connected_edge_meshgrid(new_x, new_y, connected_edge_meshgrid, Ori)
        inspected += 1

    # format output
    E = make_output_binary(M, strong_edges_x, strong_edges_y)
    return E


# for threshold
def find_strong_and_super_weak(M, Mag):
    
    '''
        A much better way to do thresholding is to create a function that considers the
        magnitude of all pixels in the photo via binning, determine the distribution, and
        choose a set standard deviation away to include the appropriate number of pixels. And
        then set the upper threshold and lower thresholds accordingly. Even better would be writing
        some sort of local adaptive threshold method
    '''
    
    # set values for thresholding
    high_threshold = 2.675  # determined experimentally using FIJI
    low_threshold = 1.00  # determined experimentally using FIJI
    
    # from set of edges, find strong and super weak edges
    [strong_edges_x, strong_edges_y] = np.where((Mag > high_threshold) & (M == 1))
    [super_weak_x, super_weak_y] = np.where((Mag < low_threshold) & (M == 1))
    
    return strong_edges_x, strong_edges_y, super_weak_x, super_weak_y


# for some blob analysis
def generate_padded_meshgrid(matrix):
    [height, width] = matrix.shape
    # 8 depth because there are eight adjacent points to the one in consideration
    meshgrid = np.empty((8, height, width))
    meshgrid[0, :, :] = np.pad(matrix, ((1, 0), (1, 0)), mode='constant')[:-1, :-1]
    meshgrid[1, :, :] = np.pad(matrix, ((1, 0), (0, 0)), mode='constant')[:-1, :]
    meshgrid[2, :, :] = np.pad(matrix, ((1, 0), (0, 1)), mode='constant')[:-1, 1:]
    meshgrid[3, :, :] = np.pad(matrix, ((0, 0), (1, 0)), mode='constant')[:, :-1]
    meshgrid[4, :, :] = np.pad(matrix, ((0, 0), (0, 1)), mode='constant')[:, 1:]
    meshgrid[5, :, :] = np.pad(matrix, ((0, 1), (1, 0)), mode='constant')[1:, :-1]
    meshgrid[6, :, :] = np.pad(matrix, ((0, 1), (0, 0)), mode='constant')[1:, :]
    meshgrid[7, :, :] = np.pad(matrix, ((0, 1), (0, 1)), mode='constant')[1:, 1:]
    return meshgrid


def meshgrid_shift(x, y, index, strong_edges_x, strong_edges_y, count, meshgrid):
    if index == 0:
        new_x = x - 1
        new_y = y - 1
    if index == 1:
        new_x = x
        new_y = y - 1
    if index == 2:
        new_x = x + 1
        new_y = y - 1
    if index == 3:
        new_x = x - 1
        new_y = y
    if index == 4:
        new_x = x + 1
        new_y = y
    if index == 5:
        new_x = x - 1
        new_y = y + 1
    if index == 6:
        new_x = x
        new_y = y + 1
    if index == 7:
        new_x = x + 1
        new_y = y + 1
    [width, height] = meshgrid.shape
    if new_x < width & new_y < height:
        strong_edges_x.append(new_x)
        strong_edges_y.append(new_y)
        count += 1
    return new_x, new_y


def meshgrid_unshift(new_x, new_y, index):
    if index == 0:
        old_x = new_x + 1
        old_y = new_y + 1
    if index == 1:
        old_x = new_x
        old_y = new_y + 1
    if index == 2:
        old_x = new_x - 1
        old_y = new_y + 1
    if index == 3:
        old_x = new_x + 1
        old_y = new_y
    if index == 4:
        old_x = new_x - 1
        old_y = new_y
    if index == 5:
        old_x = new_x + 1
        old_y = new_y - 1
    if index == 6:
        old_x = new_x
        old_y = new_y - 1
    if index == 7:
        old_x = new_x - 1
        old_y = new_y - 1
    return old_x, old_y


def update_connected_edge_meshgrid(new_x, new_y, meshgrid, ori):
    [depth, width, height] = meshgrid.shape
    for i in range(0, 8):
        [old_x, old_y] = meshgrid_unshift(new_x, new_y, i)
        if ((old_y < height) & (old_y >= 0)) & ((old_x < width) & (old_x >= 0)):
            if (abs(ori[old_x, old_y] - ori[new_x, new_y]) < .375) | (abs(ori[old_x, old_y] - ori[new_x, new_y]) > 5.90):
                meshgrid[i, old_x, old_y] = 0;
    return meshgrid


def make_output_binary(m, strong_edges_x, strong_edges_y):
    [width, height] = m.shape
    E = np.zeros((width, height), dtype=np.bool)
    E[strong_edges_x, strong_edges_y] = True
    return E
