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

def getN(img,target, angle):


	imgN = img
	center = [target[1],target[0]];


	if(center[1] < 300 or center[1] > 900 or center[0] < 300 or center[0] > 700):
	    return imgN, 0.0, 0.0 ,0.0

	angle1 = 1.9 - angle
	angle2 = 3.65 - angle
	angle3 = 0.37 - angle

	point2 = [int(round(center[0]-190*cos(angle1))), int(round(center[1]+190*sin(angle1)))]
	point3 = [int(round(center[0]-290*cos(angle2))), int(round(center[1]+290*sin(angle2)))]
	point4 = [int(round(center[0]-300*cos(angle3))), int(round(center[1]+300*sin(angle3)))]

	img[center[0],center[1]] = 50;
	img[point2[0],point2[1]] = 50;
	img[point3[0],point3[1]] = 50;
	img[point4[0],point4[1]] = 50;

	bottom_left_hor = -10;
	bottom_left_ver = -10;
	bottom_right_ver = -10;
	bottom_right_hor = -10;

	angleout = 0;
	newy = 0;
	newx = 0;

	state = 0
	center = point2
	while center[0] < 948:
		if state == 0:
			if  img[center[0]+1,center[1]] == 0:
				state = 1
		elif state == 1:
			if img[center[0]+1, center[1]] >= 240:
				break
			img[center[0]+1, center[1]] = 99;
			bottom_right_ver = bottom_right_ver + 1;
		center = [center[0]+1,center[1]]
	state = 0

	center = point2
	while center[1] < 1278:
		if state == 0:
			if img[center[0],center[1]+1] == 0:
				state = 1
		elif state == 1:
			if img[center[0],center[1]+1] >= 240:
				break
			img[center[0],center[1]] = 99;
			bottom_right_hor = bottom_right_hor + 1;
		center = [center[0],center[1]+1]

	state = 0
	center = point3

	while center[0] < 948:
		if state == 0:
			if img[center[0]+1,center[1]] == 0:
				state = 1
		elif state == 1:

			if img[center[0]+1, center[1]] >= 240:
				break
			img[center[0]+1,center[1]] = 99;
			bottom_left_ver = bottom_left_ver + 1;
		center = [center[0]+1,center[1]]

	print(bottom_left_ver)
	print(bottom_right_ver)
	print(bottom_right_hor)


	angle2 = (np.arcsin((bottom_right_ver-bottom_left_ver)*0.085/30))
	newy = ((bottom_left_ver + bottom_right_ver) / 2)
	newx = (-bottom_right_hor + (newy)*0.4)

	print("Angle, y, x")
	print(angle2)
	print(newy)
	print(newx)
	
	if(angleout > 0.3 or angleout < -0.3):
	
		print("Fabric Orientation Error")

	imgN = img
	return imgN,angle2, newx, newy	


cap = cv.VideoCapture(0)

# cap.set(cv.CAP_PROP_FRAME_WIDTH,4208)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT,3120)

# cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

while(True):

	ret, src = cap.read()

	#src = cv.resize(src,(1280,949))

	if cv.waitKey(1) & 0xFF == ord('q'):
	    break

	# Check if image is loaded successfully
	if src is None:
	    print('Could not open or find the image: ', args.input)
	    exit(0)

	# Convert image to grayscale
	blue = src.copy()
	blue[:, :, 1] = 0
	blue[:, :, 2] = 0

	_ , bigblue = cv.threshold(blue, 160, 255, cv.THRESH_BINARY)

	red = src.copy()
	red[:, :, 1] = 0
	red[:, :, 0] = 0

	_ , bigred = cv.threshold(red, 160, 255, cv.THRESH_BINARY)

	green = src.copy()
	green[:, :, 2] = 0
	green[:, :, 0] = 0

	_ , biggreen = cv.threshold(green, 160, 255, cv.THRESH_BINARY)
	_ , littlered = cv.threshold(red, 130, 255, cv.THRESH_BINARY)
	_ , littlegreen = cv.threshold(green, 170, 255, cv.THRESH_BINARY)
	greyedgreen = cv.cvtColor(green, cv.COLOR_BGR2GRAY)
	greyedred = cv.cvtColor(red, cv.COLOR_BGR2GRAY)
	greyedblue = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)


	_, bigyellow = cv.threshold(greyedgreen + greyedred - greyedblue*2, 110, 255, cv.THRESH_BINARY)
	_, bigcyan = cv.threshold(greyedgreen + greyedblue - greyedred*1.5, 50, 255, cv.THRESH_BINARY)
	_, biggreen = cv.threshold(greyedgreen*0.4 - greyedred - greyedblue*0.4, 115, 255, cv.THRESH_BINARY)
	_, bigred = cv.threshold(-greyedgreen + greyedred*2 - greyedblue, 115, 255, cv.THRESH_BINARY)
	#bigred = (greyedred - greyedblue/2 - greyedgreen/2)
	bigred = bigred.clip(min=0)

	# bigyellow = cv.cvtColor(bigyellow, cv.COLOR_BGR2GRAY)
	# _, bigyellow = cv.threshold(bigyellow, 200, 255, cv.THRESH_BINARY)


	gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

	average = int(np.mean(gray))
	if average == 0:
		average = 115
	adjustment = 1 - (115 / average)
	#print(average)
	#print(adjustment)

	# Corrects for Brightness
	fixed = cv.add(gray,np.array(adjustment))


	# print(np.shape(gray))
	# print(np.shape(fixed))

	# Convert image to binary



	_ , bigwhite = cv.threshold(gray, 150, 255, cv.THRESH_BINARY) ## Will try and correct for lighting:  + cv.THRESH_OTSU

	#contours , _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
	center = 0    

	# for i, c in enumerate(contours):
	#     # Calculate the area of each contour
	#     area = cv.contourArea(c);
	#     # Ignore contours that are too small or too large
	    
	#     # Project
	#     #if area > 60000 and area < 100000 :
	#         #print(area)
	#     if area < 70000 or 1000000 < area:
	#         continue

	#     # Draw each contour only for visualisation purposes
	    
	#     # Find the orientation of each shape
	#     angle, center = getOrientation(c, src)    

	#     if(center[1] < 200 or center[1] > 900 or center[0] < 200 or center[0] > 900):
	#         continue

	#     #print(center)
	#     cv.drawContours(src, contours, i, (0, 0, 255), 2);

	#     if center != 0:
	#         bw,angle2,newx,newy = getN(bw,center, angle)
	    
	#     #bw[center[1],center[0]] = 100
	#     position = Float32MultiArray()
	#     if center != 0:Za	
	#         position.data = [center[0],center[1],angle,angle2,newx,newy]
	#         posepub.publish(position)


	cv.imshow('Original',src)
	cv.imshow('Cyan', bigcyan)
	cv.imshow('Yellow', bigyellow)
	cv.imshow('Green', biggreen)
	cv.imshow('White',bigwhite - bigyellow)

	#cv.imshow('Black and White', bw)

cap.release()
cv.destroyAllWindows()

