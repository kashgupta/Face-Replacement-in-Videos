from faceReplacement import faceReplacement
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1File = "Data/Easy/TheMartian0.jpg"
img2File = "Data/Easy/MarquesBrownlee0.jpg"

img1=mpimg.imread(img1File)
img2=mpimg.imread(img2File)

[newImg1, newImg2] = faceReplacement(img1, img2)