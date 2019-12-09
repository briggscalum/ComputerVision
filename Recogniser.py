from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np

from math import atan2, cos, sin, sqrt, pi
import time

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)


    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    


    mean, eigenvectors = cv.PCACompute(data_pts, mean)
    
    # eigenvalues, eigenvectors = np.linalg.eig(eigenvectors)

    #eigenvalues = cv.eigen(eigenvectors)
    #np.linalg.eig

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] , cntr[1] + 0.02 * eigenvectors[0,1] )
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] , cntr[1] - 0.02 * eigenvectors[1,1])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return angle, cntr



cap = cv.VideoCapture(0)

# cap.set(cv.CAP_PROP_FRAME_WIDTH,4208)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT,3120)
# cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

src = cv.imread('bird0_R67_S1-95_M10.png',cv.IMREAD_COLOR)

#while(True):

	#ret, src = cap.read()
	#print("foo")
	
srcgrey = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret0, thresh = cv.threshold(srcgrey, 127, 255, 0)

#src = cv.resize(src,(1280,949))

# if cv.waitKey(1) & 0xFF == ord('q'):
#     break

# Check if image is loaded successfully
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

# Convert image to grayscale
blue = src.copy()
blue[:, :, 1] = 1
blue[:, :, 2] = 1


preblue = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)
_ , bigblue = cv.threshold(preblue, 10, 255, cv.THRESH_BINARY)

red = src.copy()
red[:, :, 1] = 1
red[:, :, 0] = 1

prered =cv.cvtColor(red, cv.COLOR_BGR2GRAY)
_ , bigred = cv.threshold(prered, 160, 255, cv.THRESH_BINARY)

green = src.copy()
green[:, :, 2] = 1
green[:, :, 0] = 1

pregreen =cv.cvtColor(green, cv.COLOR_BGR2GRAY)
_ , biggreen = cv.threshold(pregreen, 160, 255, cv.THRESH_BINARY)
_ , littlered = cv.threshold(red, 130, 255, cv.THRESH_BINARY)
_ , littlegreen = cv.threshold(green, 170, 255, cv.THRESH_BINARY)
greyedgreen = cv.cvtColor(green, cv.COLOR_BGR2GRAY)
greyedred = cv.cvtColor(red, cv.COLOR_BGR2GRAY)
greyedblue = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)


_, bigyellow = cv.threshold(pregreen + prered - preblue*4, 130, 255, cv.THRESH_BINARY)
precyan = pregreen + preblue
for i in range (len(precyan)):
	for j in range (len(precyan[i])):
		if(precyan[i][j] > prered[i][j]*2.5):
			precyan[i][j] = precyan[i][j] - prered[i][j]*2.5
		else:
			precyan[i][j] = 0
#precyan = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)
ret, bigcyan = cv.threshold(precyan, 20, 255, 0)

_, biggreen = cv.threshold(greyedgreen*0.4 - greyedred - greyedblue*0.4, 115, 255, cv.THRESH_BINARY)
_, bigred = cv.threshold(-greyedgreen + greyedred*2 - greyedblue, 115, 255, cv.THRESH_BINARY)

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

average = int(np.mean(gray))
if average == 0:
	average = 115
adjustment = 1 - (115 / average)


# Corrects for Brightness
fixed = cv.add(gray,np.array(adjustment))
_ , bigwhite = cv.threshold(gray, 200, 255, cv.THRESH_BINARY) ## Will try and correct for lighting:  + cv.THRESH_OTSU
invert = -gray
_ , bigblack = cv.threshold(invert, 180, 255, cv.THRESH_BINARY) ## Will try and correct for lighting:  + cv.THRESH_OTSU

contours , _ = cv.findContours(bigcyan, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
ycontours , _ = cv.findContours(bigyellow, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
wcontours , _ = cv.findContours(bigwhite, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
bcontours , _ = cv.findContours(bigblack, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#center = 0    
for i, c in enumerate(bcontours):
	area = cv.contourArea(c);
	cv.drawContours(src, bcontours, i, (0, 0, 0), 2);
for i, c in enumerate(ycontours):
	area = cv.contourArea(c);
	cv.drawContours(src, ycontours, i, (0, 255, 255), 2);
for i, c in enumerate(wcontours):
	area = cv.contourArea(c);
	cv.drawContours(src, wcontours, i, (255, 255, 255), 2);

for i, c in enumerate(contours):
	# Calculate the area of each contour
	area = cv.contourArea(c);
	# Ignore contours that are too small or too large

	# Project
	#if area > 60000 and area < 100000 :
	    #print(area)
	# if area < 70000 or 1000000 < area:
	#     continue

	# Draw each contour only for visualisation purposes

	# Find the orientation of each shape
	# angle, center = getOrientation(c, src)    

	# if(center[1] < 200 or center[1] > 900 or center[0] < 200 or center[0] > 900):
	#     continue

	#print(center)
	cv.drawContours(src, contours, i, (255, 0, 0), 2);

	# if center != 0:
	#     bw,angle2,newx,newy = getN(bw,center, angle)

	#bw[center[1],center[0]] = 100

cv.imshow('Original',src)
cv.imshow('cyan', bigcyan)

cv.imshow('Green', biggreen)
cv.imshow('White',bigwhite - bigyellow)
cv.imshow('Black',bigblack)
cv.imshow('Yellow', bigyellow)
cv.waitKey(0)

	#cv.imshow('Black and White', bw)

cap.release()
cv.destroyAllWindows()

