from __future__ import print_function
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# BUG: if sum or diff of two points is the same, wrong indexes could be chosen for corners
def order_points_old(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is in the top-left
    # the second entry is in the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4,2),dtype='float32')

    # top-left point will have the smallest sum, whereas
    # the bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help='path to input image')
ap.add_argument('-n','--new',type=int,default=-1,help='whether or not the new order points should be used.')
args = vars(ap.parse_args())

# load the input image, convert it to grayscale, and blur it slightly
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(7,7),0)

# perform Canny edge detection, then perform a dilation + erosion to close gaps in between object edges
edged = cv2.Canny(gray,50,100)
edged = cv2.dilate(edged,None,iterations=1)
edged = cv2.erode(edged,None,iterations=1)