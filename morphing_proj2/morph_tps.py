'''
  File name: morph_tps.py
  Author: Rajiv Patel-O'Connor
  Date created: 10-18-17
'''

'''
  File clarification:
    Image morphing via TPS
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from est_tps import est_tps
from obtain_morphed_tps import obtain_morphed_tps

def morph_tps(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  # TODO: Your code here
  # May do as extra-credit but not now
  return morphed_im
