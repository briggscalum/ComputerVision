from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np

from math import atan2, cos, sin, sqrt, pi
import time

    
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors = cv.PCACompute(data_pts, mean)
    cntr = (int(mean[0,0]), int(mean[0,1]))
    cv.circle(img, cntr, 3, (255, 255, 255), 2)
    angle = 1
    return angle, cntr

cap = cv.VideoCapture(0)



src = cv.imread('upside.jpg',cv.IMREAD_COLOR)

	
srcgrey = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret0, thresh = cv.threshold(srcgrey, 127, 255, 0)


# Check if image is loaded successfully
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

# Generate greyscale images of each color

blue = src.copy()
blue[:, :, 1] = 1
blue[:, :, 2] = 1
preblue = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)

red = src.copy()
red[:, :, 1] = 1
red[:, :, 0] = 1
prered =cv.cvtColor(red, cv.COLOR_BGR2GRAY)

green = src.copy()
green[:, :, 2] = 1
green[:, :, 0] = 1
pregreen =cv.cvtColor(green, cv.COLOR_BGR2GRAY)

preyellow = pregreen + prered 
precyan = pregreen + preblue*2

for i in range (len(precyan)):
	for j in range (len(precyan[i])):
		if(preyellow[i][j] > preblue[i][j]*6):
			preyellow[i][j] = preyellow[i][j] - preblue[i][j]*6
		else:
			preyellow[i][j] = 0
		if(precyan[i][j] > prered[i][j]*2.5):
			precyan[i][j] = precyan[i][j] - prered[i][j]*2.5
		else:
			precyan[i][j] = 0

ret, bigcyan = cv.threshold(precyan, 40, 255, 0)
ret, bigyellow = cv.threshold(preyellow, 100, 255, 0)
gray = src.copy()
og = src.copy()

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
gray = gray

average = int(np.mean(gray))
if average == 0:
	average = 115
adjustment = 1 - (115 / average)


# Corrects for Brightness
fixed = cv.add(gray,np.array(adjustment))
_ , bigwhite = cv.threshold(gray - preyellow , 150, 255, cv.THRESH_BINARY) ## Will try and correct for lighting:  + cv.THRESH_OTSU
invert = -gray
_ , bigblack = cv.threshold(invert, 210, 255, cv.THRESH_BINARY) ## Will try and correct for lighting:  + cv.THRESH_OTSU

contours , _ = cv.findContours(bigcyan, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
ycontours , _ = cv.findContours(bigyellow, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
bigwhiter = bigwhite - bigyellow
wcontours , _ = cv.findContours(bigwhite, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
bcontours , _ = cv.findContours(bigblack, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#center = 0    

maxarea = 0
angle = 0
center = 0

# Find largest Blue Contour

for i, c in enumerate(contours):
	# Calculate the area of each contour
	area = cv.contourArea(c)
	if area > maxarea:
		maxarea = area
bluecenter = 0

# Draw Blue Contour

for i, c in enumerate(contours):
	# Calculate the area of each contour
	area = cv.contourArea(c)
	if area == maxarea:
		cv.drawContours(src, contours, i, (255, 0, 0), 2)
		angle, center = getOrientation(c, src) 
		bluecenter = center
		#area = area*2
		print(center)

# Draw Black Contours

for i, c in enumerate(bcontours):
	#angle, center = getOrientation(c, src)
	area = cv.contourArea(c);
	if area > maxarea/100:
	    continue
	cv.drawContours(src, bcontours, i, (0, 0, 0), 2);


# Draw Yellow Contours

yellowcount = 0
yellowcenter = [0,0]
for i, c in enumerate(ycontours):
	area = cv.contourArea(c);
	if area > maxarea/3 or area < maxarea/30 :
	    continue
	angle, center = getOrientation(c, bigyellow)
	if((center[0] - bluecenter[0])**2 + (center[1] - bluecenter[1])**2 > maxarea*3):
		continue
	yellowcenter[0] = yellowcenter[0] + center[0]
	yellowcenter[1] = yellowcenter[1] + center[1]
	yellowcount = yellowcount + 1
	area = cv.contourArea(c)
	cv.drawContours(src, ycontours, i, (0, 255, 255), 2)
yellowcenter = (int(yellowcenter[0]/yellowcount),int(yellowcenter[1]/yellowcount))


# Draw White Contours

closest = 1000000
closestcontour = 0
closesti = 0
for i, c in enumerate(wcontours):
	area = cv.contourArea(c);
	if area > maxarea/2 or area < maxarea/5:
		continue
	angle, center = getOrientation(c, src) 
	distance = (center[0] - bluecenter[0])**2 + (center[1] - bluecenter[1])**2
	if distance < closest:
		closest = distance
		closestcontour = c
		closesti = i


#Draw bounding box

angle, whitecenter = getOrientation(closestcontour, src) 
print(whitecenter)
cv.drawContours(src, wcontours, closesti, (200, 200, 200), 2);
cv.line(src, (bluecenter[0],bluecenter[1]), (yellowcenter[0], yellowcenter[1]), (50,100,100), 5)
cv.line(src, (whitecenter[0],whitecenter[1]), (bluecenter[0], bluecenter[1]), (50,50,50), 5)	
maxarea = maxarea*0.5

# Draw Skeleton

if(whitecenter[1] < yellowcenter[1]):
	cv.line(src, (whitecenter[0] - int(sqrt(maxarea)), whitecenter[1] - int(sqrt(maxarea))), ((yellowcenter[0] - int(sqrt(maxarea))), (yellowcenter[1] + int(sqrt(maxarea)))), (0,255,0), 2)
	cv.line(src, (yellowcenter[0] - int(sqrt(maxarea)), yellowcenter[1] + int(sqrt(maxarea))), ((yellowcenter[0] + int(sqrt(maxarea))), (yellowcenter[1] + int(sqrt(maxarea)))), (0,255,0), 2)
	cv.line(src, (yellowcenter[0] + int(sqrt(maxarea)), yellowcenter[1] + int(sqrt(maxarea))), ((whitecenter[0] + int(sqrt(maxarea))), (whitecenter[1] - int(sqrt(maxarea)))), (0,255,0), 2)
	cv.line(src, (whitecenter[0] + int(sqrt(maxarea)), whitecenter[1] - int(sqrt(maxarea))), ((whitecenter[0] - int(sqrt(maxarea))), (whitecenter[1] - int(sqrt(maxarea)))), (0,255,0), 2)
else:
	cv.line(src, (yellowcenter[0] - int(sqrt(maxarea)), yellowcenter[1] - int(sqrt(maxarea))), ((whitecenter[0] - int(sqrt(maxarea))), (whitecenter[1] + int(sqrt(maxarea)))), (0,255,0), 2)
	cv.line(src, (whitecenter[0] - int(sqrt(maxarea)), whitecenter[1] + int(sqrt(maxarea))), ((whitecenter[0] + int(sqrt(maxarea))), (whitecenter[1] + int(sqrt(maxarea)))), (0,255,0), 2)
	cv.line(src, (whitecenter[0] + int(sqrt(maxarea)), whitecenter[1] + int(sqrt(maxarea))), ((yellowcenter[0] + int(sqrt(maxarea))), (yellowcenter[1] - int(sqrt(maxarea)))), (0,255,0), 2)
	cv.line(src, (yellowcenter[0] + int(sqrt(maxarea)), yellowcenter[1] - int(sqrt(maxarea))), ((yellowcenter[0] - int(sqrt(maxarea))), (yellowcenter[1] - int(sqrt(maxarea)))), (0,255,0), 2)

# Display Images

cv.imshow('Original', og)
cv.imshow('Cyan', precyan)
cv.imshow('White', gray - preyellow)
cv.imshow('Black',invert)
cv.imshow('Yellow', bigyellow)
cv.imshow('Final',src)
cv.waitKey(0)

	#cv.imshow('Black and White', bw)

cap.release()
cv.destroyAllWindows()

