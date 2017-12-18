import cv2
import numpy as np

# Warps and alpha blends triangular regions from img1 and img2
def warpTriangle(img1, img2, tri1, tri2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([tri1]))
    r2 = cv2.boundingRect(np.float32([tri2]))

    # Offset points by left top corner of the respective rectangles
    tri1Rect = np.zeros((3,2))
    tri2Rect = np.zeros((3,2))
    tri2RectInt = np.zeros((3,2))

    #iterate over the three points
    for i in range(0, 3):
        tri1Rect[i] = ((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1]))
        tri2Rect[i] = (((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))
        tri2RectInt[i] = (((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))

    # Get mask
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to rectangles
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    # get the size of the rectangle
    size = (r2[2], r2[3])

    #do the affine transformation
    warpMat = cv2.getAffineTransform(np.float32(tri1Rect), np.float32(tri2Rect))
    img2Rect = cv2.warpAffine(img1Rect, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

    #get the masked rectangle
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect