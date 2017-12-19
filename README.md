# Face-Replacement-in-Videos

You need the following packages:
* cv2 (OpenCV)
* numpy
* dlib
* skvideo.io (pip install sk-video)
* matplotlib.pyplot
* scipy.spatial.Delaunay

To run, simply put in the file paths for the two videos into test.py (our code only handles one face per video).

In console you should see which frame the algorithm is analyzing.

The output videos will be named as follows face2_on_video1 and face1_on_video2. If you would like to change this, you can change this in faceReplacementForVideo in the lines that contain skvidoe.io.vrite().
