# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")

ap.add_argument("-i", "--image", required=True,
	help="path to input image")

args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

####

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect faces in the grayscale image
rects = detector(gray, 1)

####

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
 
	
	#eye to eye = p1	
	p1 = math.sqrt(((shape[36,0] - shape[45,0])**2) + ((shape[36,1] - shape[45,1])**2))
	
	#eye to nose bridge = p2	
	p2 = math.sqrt(((shape[36,0] - shape[27,0])**2) + ((shape[36,1] - shape[27,1])**2))

	#nose bridge to nose tip = p3
	p3 = math.sqrt(((shape[27,0] - shape[30,0])**2) + ((shape[27,1] - shape[30,1])**2))

	#nose tip to chin = p4
	p4 = math.sqrt(((shape[30,0] - shape[8,0])**2) + ((shape[30,1] - shape[8,1])**2))

	#nose bridge to chin = p5
	p5 = math.sqrt(((shape[27,0] - shape[8,0])**2) + ((shape[27,1] - shape[8,1])**2))

	#chin to eye = p6
	p6 = math.sqrt(((shape[8,0] - shape[36,0])**2) + ((shape[8,1] - shape[36,1])**2))

	#eyebrow length = p7
	p7 = math.sqrt(((shape[17,0] - shape[21,0])**2) + ((shape[17,1] - shape[21,1])**2))

	#nose tip to left eyebrow = p8
	p8 = math.sqrt(((shape[30,0] - shape[17,0])**2) + ((shape[30,1] - shape[17,1])**2))

	#nose tip to right eyebrow = p9
	p9 = math.sqrt(((shape[30,0] - shape[45,0])**2) + ((shape[30,1] - shape[45,1])**2))

	#overall length = p10
	p10 = math.sqrt(((shape[0,0] - shape[16,0])**2) + ((shape[0,1] - shape[16,1])**2))
		
 	ans = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
	print ans
# show the output image with the face detections + facial landmarks
#cv2.imshow("Output", image)
cv2.waitKey(0)