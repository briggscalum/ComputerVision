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
    
    
    cv.circle(img, cntr, 3, (255, 255, 255), 2)
    # p1 = (cntr[0] + 0.02 * eigenvectors[0,0] , cntr[1] + 0.02 * eigenvectors[0,1] )
    # p2 = (cntr[0] - 0.02 * eigenvectors[1,0] , cntr[1] - 0.02 * eigenvectors[1,1])
    # drawAxis(img, cntr, p1, (0, 255, 0), 1)
    # drawAxis(img, cntr, p2, (255, 255, 0), 5)
    # angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    angle = 1
    return angle, cntr



cap = cv.VideoCapture(0)

# cap.set(cv.CAP_PROP_FRAME_WIDTH,4208)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT,3120)
# cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

src = cv.imread('Duck3.jpg',cv.IMREAD_COLOR)

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


preyellow = pregreen + prered 
precyan = pregreen + preblue*2


for i in range (len(precyan)):
	for j in range (len(precyan[i])):
		if(preyellow[i][j] > preblue[i][j]*4):
			preyellow[i][j] = preyellow[i][j] - preblue[i][j]*4
		else:
			preyellow[i][j] = 0
		if(precyan[i][j] > prered[i][j]*2.5):
			precyan[i][j] = precyan[i][j] - prered[i][j]*2.5
		else:
			precyan[i][j] = 0
#precyan = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)
ret, bigcyan = cv.threshold(precyan, 40, 255, 0)
ret, bigyellow = cv.threshold(preyellow, 100, 255, 0)
# _, bigyellow = cv.threshold(pregreen + prered - preblue*4, 130, 255, cv.THRESH_BINARY)
_, biggreen = cv.threshold(greyedgreen*0.4 - greyedred - greyedblue*0.4, 115, 255, cv.THRESH_BINARY)
_, bigred = cv.threshold(-greyedgreen + greyedred*2 - greyedblue, 115, 255, cv.THRESH_BINARY)


gray = src.copy()

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
gray = gray

average = int(np.mean(gray))
if average == 0:
	average = 115
adjustment = 1 - (115 / average)


# Corrects for Brightness
fixed = cv.add(gray,np.array(adjustment))
_ , bigwhite = cv.threshold(gray , 200, 255, cv.THRESH_BINARY) ## Will try and correct for lighting:  + cv.THRESH_OTSU
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

for i, c in enumerate(contours):
	# Calculate the area of each contour
	area = cv.contourArea(c)
	if area > maxarea:
		maxarea = area
bluecenter = 0
for i, c in enumerate(contours):
	# Calculate the area of each contour
	area = cv.contourArea(c)
	if area == maxarea:
		cv.drawContours(src, contours, i, (255, 0, 0), 2)
		angle, center = getOrientation(c, src) 
		bluecenter = center
		area = area*2
		print(center)
		cv.line(src, (center[0] - int(sqrt(area)), center[1] - int(sqrt(area))), ((center[0] - int(sqrt(area))), (center[1] + int(sqrt(area)))), (0,255,0), 10)
		cv.line(src, (center[0] - int(sqrt(area)), center[1] + int(sqrt(area))), ((center[0] + int(sqrt(area))), (center[1] + int(sqrt(area)))), (0,255,0), 10)
		cv.line(src, (center[0] + int(sqrt(area)), center[1] + int(sqrt(area))), ((center[0] + int(sqrt(area))), (center[1] - int(sqrt(area)))), (0,255,0), 10)
		cv.line(src, (center[0] + int(sqrt(area)), center[1] - int(sqrt(area))), ((center[0] - int(sqrt(area))), (center[1] - int(sqrt(area)))), (0,255,0), 10)



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
	

	# if center != 0:
	#     bw,angle2,newx,newy = getN(bw,center, angle)

	#bw[center[1],center[0]] = 100

for i, c in enumerate(bcontours):
	#angle, center = getOrientation(c, src)
	area = cv.contourArea(c);
	if area > maxarea/100:
	    continue
	cv.drawContours(src, bcontours, i, (0, 0, 0), 2);
yellowcount = 0
yellowcenter = [0,0]
for i, c in enumerate(ycontours):
	area = cv.contourArea(c);
	if area > maxarea/3 or area < maxarea/30 :
	    continue
	angle, center = getOrientation(c, bigyellow)
	yellowcenter[0] = yellowcenter[0] + center[0]
	yellowcenter[1] = yellowcenter[1] + center[1]
	yellowcount = yellowcount + 1
	area = cv.contourArea(c);
	cv.drawContours(src, ycontours, i, (0, 255, 255), 2);
#print(yellowcount)
yellowcenter = (int(yellowcenter[0]/yellowcount),int(yellowcenter[1]/yellowcount))
# cv.circle(src, yellowcenter, 1, (0, 255, 255), 10) 

closest = 1000000
closestcontour = 0
closesti = 0
for i, c in enumerate(wcontours):
	area = cv.contourArea(c);
	if area > maxarea/1 or area < maxarea/100 :
		continue
	#cv.drawContours(src, wcontours, i, (100, 100, 100), 2);
	angle, center = getOrientation(c, src) 
	distance = (center[0] - bluecenter[0])**2 + (center[1] - bluecenter[1])**2
	#print(distance)
	if distance < closest:
		closest = distance
		closestcontour = c
		closesti = i

angle, whitecenter = getOrientation(closestcontour, src) 
print(whitecenter)
cv.drawContours(src, wcontours, closesti, (200, 200, 200), 2);

cv.line(src, (bluecenter[0],bluecenter[1]), (yellowcenter[0], yellowcenter[1]), (50,100,100), 5)
cv.line(src, (whitecenter[0],whitecenter[1]), (bluecenter[0], bluecenter[1]), (50,50,50), 5)		



cv.imshow('cyan', bigcyan)
cv.imshow('Green', biggreen)
cv.imshow('White',bigwhite)
cv.imshow('Black',bigblack)
cv.imshow('Yellow', bigyellow)
cv.imshow('grey', gray)
cv.imshow('Original',src)
cv.waitKey(0)

	#cv.imshow('Black and White', bw)

cap.release()
cv.destroyAllWindows()

