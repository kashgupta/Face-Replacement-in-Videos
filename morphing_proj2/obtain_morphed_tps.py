'''
  File name: obtain_morphed_tps.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing based on TPS parameters 
    - Input im_source: the source image
    - Input a1_x, ax_x, ay_x, w_x: the TPS parameters in x dimension
    - Input a1_y, ax_y, ay_y, w_y: the TPS parameters in y dimension
    - Input interim_pts: correspondences position in the intermediate image 
    - Input sz: the vector contains the intermediate image size

    - Output morphed_im: the morphed image, the output size should be [sz[0], sz[1], 3]
'''

def obtain_morphed_tps(im_source, a1_x, ax_x, ay_x, w_x, a1_y, ax_y, ay_y, w_y, interim_pts, sz):
  # TODO: Your code here

  return morphed_im