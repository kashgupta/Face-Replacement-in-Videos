import cv2
from scipy.spatial import Delaunay
import numpy as np
from facialLandmark import facialLandmark
from warpTriangle import warpTriangle

def faceReplacement(img1File, img2File):
    img1 = cv2.imread(img1File);
    img2 = cv2.imread(img2File);
    img1Warped = np.copy(img2);

    #find the landmarks
    landmarks1 = facialLandmark(img1)
    landmarks2 = facialLandmark(img2)

    #get the convex hull
    indices = cv2.convexHull(np.array(landmarks2), returnPoints = False)
    hull1 = landmarks1[indices[:,0]]
    hull2 = landmarks2[indices[:,0]]
    #convert hulls to tuple list
    hull1 = tuple(map(tuple, hull1))
    hull2 = tuple(map(tuple, hull2))

    #create delanuy triangulation
    tri = Delaunay(hull2).simplices
    #conver to tuple list
    tri = tuple(map(tuple, tri))

    #for each of the triangles, do the affine warp
    numTriangles = len(tri)
    for i in range(0, numTriangles):
        triangle = tri[i]
        pts1 = np.zeros((3, 2), dtype=np.float32)
        pts2 = np.zeros((3, 2), dtype=np.float32)

        # calculate transform for each triangle in both directions
        pts1[0] = hull1[triangle[0]]
        pts1[1] = hull1[triangle[1]]
        pts1[2] = hull1[triangle[2]]

        pts2[0] = hull2[triangle[0]]
        pts2[1] = hull2[triangle[1]]
        pts2[2] = hull2[triangle[2]]

        warpTriangle(img1, img1Warped, pts1, pts2)

    #get the mask of only the hull for blending
    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))

    #find the center of the face to do the blending
    r = cv2.boundingRect(np.float32([hull2]))
    faceCenter = (r[0]+int(r[2]/2), r[1]+int(r[3]/2))

    #do the blending
    finalImage = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, faceCenter, cv2.NORMAL_CLONE)

    #show the reesult
    cv2.imshow('image', finalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # #do face detection and get bounding boxes
    # #returns numFacesx4x2 where 4 is four corners and 3rd dimension is xy
    # bboxImg1 = detectFace(img1)
    # bboxImg2 = detectFace(img2)
    #
    # #get facial landmarks from dlibs
    # #nx2 each
    # landmarksImg1 = facialLandmark(img1)
    # landmarksImg2 = facialLandmark(img2)
    #
    # #get the convex hulls
    # hull1 = ConvexHull(landmarksImg1)
    # hull2 = ConvexHull(landmarksImg2)
    #
    # #plot the image
    # #plt.imshow(img1)
    # #plot the triangulation
    # #plt.triplot(landmarksImg1[:, 0], landmarksImg1[:, 1], tri1.simplices.copy())
    # #plot the hull
    # #plt.plot(landmarksImg1[hull1.vertices, 0], landmarksImg1[hull1.vertices, 1], 'r--', lw=2)
    # #plt.show()
    #
    # #append the facial landmarks and the convexhulls together
    # img1Features = np.vstack((landmarksImg1,hull1.points[hull1.vertices]))
    # img2Features = np.vstack((landmarksImg2,hull2.points[hull2.vertices]))
    #
    # #make sure features same size
    # minPts = np.min([img1Features.shape[0],img2Features.shape[0]])
    # img1Features = img1Features[0:minPts,:]
    # img2Features = img2Features[0:minPts,:]
    #
    # #create delanuy triangulation
    # tri1 = Delaunay(img1Features).simplices
    # tri2 = Delaunay(img2Features).simplices
    #
    # minPtsTri = np.min([tri1.shape[0],tri2.shape[0]])
    # tri1 = tri1[0:minPtsTri,:]
    # tri2 = tri2[0:minPtsTri,:]
    #
    # [numTriangles1,_,] = tri1.shape
    # [numTriangles2,_,] = tri2.shape
    #
    # #print (numTriangles1, numTriangles2)
    #
    # #img2 = 255 * np.ones(img_in.shape, dtype=img_in.dtype)
    #
    # #allocate memory
    # loAffineTransforms12 = np.zeros((len(tri1), 2, 3)) #store affine for 1 -> 2
    # loAffineTransforms21 = np.zeros((len(tri1), 2, 3)) #store affine for 2 -> 1
    # pts1 = np.zeros((3,2), dtype=np.float32)
    # pts2 = np.zeros((3,2), dtype=np.float32)
    #
    # #calculate transform for each triangle in both directions
    # for i in range(len(tri1)):
    #     pts1[0] = img1Features[tri1[i][0]]
    #     pts1[1] = img1Features[tri1[i][1]]
    #     pts1[2] = img1Features[tri1[i][2]]
    #
    #     pts2[0] = img2Features[tri2[i][0]]
    #     pts2[1] = img2Features[tri2[i][1]]
    #     pts2[2] = img2Features[tri2[i][2]]
    #
    #     loAffineTransforms12[i] = cv2.getAffineTransform(pts1, pts2)
    #     loAffineTransforms21[i] = cv2.getAffineTransform(pts2, pts1)
    #
    #     r1 = cv2.boundingRect(pts1)
    #     r2 = cv2.boundingRect(pts2)
    #
    #     # Offset points by left top corner of the respective rectangles
    #     t1Rect = []
    #     t2Rect = []
    #
    #     for i in range(0, 3):
    #         t1Rect.append(((pts1[i][0] - r1[0]),(pts1[i][1] - r1[1])))
    #         t2Rect.append(((pts2[i][0] - r2[0]),(pts2[i][1] - r2[1])))
    #
    #     # Get mask by filling triangle
    #     mask = np.zeros((r1[3], r1[2], 3), dtype = np.float32)
    #     cv2.fillConvexPoly(mask, np.int32(t1Rect), (1.0, 1.0, 1.0), 16, 0);
    #     plt.imshow(mask)
    #
    #     # Apply warpImage to small rectangular patches
    #     img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #     img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    #
    #     #Apply Affine
    #     size1 = (r1[2], r1[3])
    #     size2 = (r2[2], r2[3])
    #     warp1 = cv2.warpAffine(img1Rect, loAffineTransforms12[i],(size1[0], size1[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    #     warp2 = cv2.warpAffine(img2Rect, loAffineTransforms21[i],(size2[0], size2[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    #
    #     # Thing at the end
    #     img2Rect = warp1 * mask
    #
    #     # Copy triangular region of the rectangular patch to the output image
    #     img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    #
    #     img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
    #
    #
    # #blending if want to
    # #we can call this function
    # #output = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)


    return [img1Warped]