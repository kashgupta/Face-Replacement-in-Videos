import cv2
import PyNet as net
from proj4AdetectFace import detectFace
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from proj2helpers import calculateAFromIdx, get_rgb
import numpy as np
from facialLandmark import facialLandmark
#from getConvexHull import getConvexHull

# img1 and img2 are nxn numpy matricies
# noinspection PyInterpreter
def faceReplacement(img1, img2):

    #do face detection and get bounding boxes
    #returns numFacesx4x2 where 4 is four corners and 3rd dimension is xy
    bboxImg1 = detectFace(img1)
    bboxImg2 = detectFace(img2)

    #get facial landmarks from the PyNet CNN
    #returns nx2 locations of the features
    # myModel = net.Model.load_model("Project4BCNN/500.pickle")
    # landmarksImg1 = myModel.forward(img1)
    # landmarksImg2 = myModel.forward(img2)

    #get facial landmarks from dlibs
    #nx2 each
    landmarksImg1 = facialLandmark(img1)
    landmarksImg2 = facialLandmark(img2)
    
    # #getRGB image
    # img1_gray = rgb2gray(img1)
    # img2_gray = rgb2gray(img2)
    #
    # #get the facial features for the image
    # #returns 250xnumFaces points for x locations and y locations
    # xFeatures1, yFeatures1 = getFeatures(img1_gray, bboxImg1)
    # features1 = np.hstack((xFeatures1,yFeatures1))
    # xFeatures2, yFeatures2 = getFeatures(img2_gray, bboxImg2)
    # features2 = np.hstack((xFeatures2,yFeatures2))
    #
    # #get the two convex hulls for the images
    # convexHull1 = cv2.convexHull(features1, returnPoints=True)
    # convexHull2 = cv2.convexHull(features2, returnPoints=True)

    #get the triangulations
    tri1 = Delaunay(landmarksImg1)
    tri2 = Delaunay(landmarksImg2)

    #get the convex hulls
    hull1 = ConvexHull(landmarksImg1)
    hull2 = ConvexHull(landmarksImg2)


    #plot the image
    #plt.imshow(img1)
    #plot the triangulation
    #plt.triplot(landmarksImg1[:, 0], landmarksImg1[:, 1], tri1.simplices.copy())
    #plot the hull
    #plt.plot(landmarksImg1[hull1.vertices, 0], landmarksImg1[hull1.vertices, 1], 'r--', lw=2)
    #plt.show()

    #append the facial landmarks and the convexhulls together
    img1Features = np.vstack((landmarksImg1,hull1.points[hull1.vertices]))
    img2Features = np.vstack((landmarksImg2,hull2.points[hull2.vertices]))

    #create delanuy triangulation
    tri1 = Delaunay(img1Features)
    tri2 = Delaunay(img2Features)

    #do affine warp of the triangles
    

    #Get simplices
    simplices1 = tri1.simplices
    simplices2 = tri2.simplices
    
        
    # Calculate A for all triangles
    loA = []
    for i in range(len(simplices1)):

        vtx1 = img1Features[simplices1[i][0]]
        vtx2 = img1Features[simplices1[i][1]]
        vtx3 = img1Features[simplices1[i][2]]
            
        A = [
            [vtx1[0], vtx2[0], vtx3[0]],
            [vtx1[1], vtx2[1], vtx3[1]],
            [1, 1, 1]
            ]
                 
        loA.append(A)
        
        #find size of bbox
        xMin = np.amin(bboxImg1[0,:,0])
        xMax = np.amax(bboxImg1[0,:,0])
        yMin = np.amin(bboxImg1[0,:,1])
        yMax = np.amax(bboxImg1[0,:,1])
        xSize = xMax - xMin
        ySize = yMax - yMin
        
        
        replacementFace = np.zeros((ySize, xSize, 3))
        for y in range(yMin, yMax):
            for x in range(xMin, xMax):
                if(tri1.find_simplex(np.array([x,y])) == -1):
                    continue
                else:
                    # Calculate barycentric coordinates
                    tri = tri1.find_simplex(np.array([x, y]))
                    alpha_beta_gamma = np.dot(np.linalg.inv(loA[tri]), np.asarray([[x], [y], [1]]))
                    
                    # Calculate corresponding coordinates
                    src_A = calculateAFromIdx(tri, img2Features, simplices2)
                    src_coord = np.dot(src_A, alpha_beta_gamma)
                    
                    # Normalization (divide by 'Z')
                    src_coord = src_coord/src_coord[2][0]
                    
                    # Interpolate to get RGB
                    src_rgb = get_rgb(src_coord[0][0], src_coord[1][0], img2)
                    
                    # Build Source Image
                    replacementFace[y - yMin][x - xMin] = src_rgb
    

    #blending if want to
    #we can call this function
    #output = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
