'''
  File name: morph_tri.py
  Author: Rajiv Patel-O'Connor
  Date created: 10-18-2017
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from scipy.spatial import Delaunay
import numpy as np
from helpers import calculateAFromIdx, get_rgb


def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):

    # initialize variables
    pics = len(warp_frac)
    h, w, d = np.shape(im1)
    morphed_im = np.empty([pics, h, w, d], dtype=np.float32)

    # calculate morphed im for given warp fractions and dissolve fractions
    for p in range(pics):
        warp = warp_frac[p]
        dissolve = dissolve_frac[p]
        
        # Generate intermediate image and do triangulation
        int_pts = im1_pts*(1-warp) + im2_pts*(warp)

        tri_int = Delaunay(int_pts)
        tri_idx_int = tri_int.simplices
        
        # Calculate A for all triangles
        list_of_A = []
        for i in range(len(tri_idx_int)):

            vtx1 = int_pts[tri_idx_int[i][0]]
            vtx2 = int_pts[tri_idx_int[i][1]]
            vtx3 = int_pts[tri_idx_int[i][2]]
            
            A = [
                 [vtx1[0], vtx2[0], vtx3[0]],
                 [vtx1[1], vtx2[1], vtx3[1]],
                 [1, 1, 1]
                 ]
                 
            list_of_A.append(A)
        
        '''
            Doing this pointwise (yes this is slow), calculate the barycentric coordinates for that point (alpha, beta, gamma)
            Then find the corresponding coordinates (in source and target) based on barycentric coordinates
            Interpolate to find RGB for each of those barycentric coordinates and stitch together to form image
        '''
        
        interpolated_im = np.empty(np.shape(im1), dtype=np.float32)
        for x in range(0, w):
            for y in range(0, h):
                
                # Calculate barycentric coordinates
                tri = tri_int.find_simplex(np.array([x, y]))
                alpha_beta_gamma = np.dot(np.linalg.inv(list_of_A[tri]), np.asarray([[x], [y], [1]]))
                
                # Calculate corresponding coordinates
                src_A = calculateAFromIdx(tri, im1_pts, tri_idx_int)
                target_A = calculateAFromIdx(tri, im2_pts, tri_idx_int)
                src_coord = np.dot(src_A, alpha_beta_gamma)
                target_coord = np.dot(target_A, alpha_beta_gamma)
                
                # Normalization (divide by 'Z')
                src_coord = src_coord/src_coord[2][0]
                target_coord = target_coord/target_coord[2][0]
                
                # Interpolate to get RGB
                src_rgb = get_rgb(src_coord[0][0], src_coord[1][0], im1)
                target_rgb = get_rgb(target_coord[0][0], target_coord[1][0], im2)
                
                # Build Source Image
                interpolated_im[y][x] = src_rgb*(1-dissolve) + target_rgb*(dissolve)
    
        morphed_im[p] = interpolated_im
    
    return morphed_im
