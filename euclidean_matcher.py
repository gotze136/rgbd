import cv2
from rgbd import rgbd
import numpy as np
import scipy
def matcher(rgb,depth):
	feature=rgbd(rgb,depth)
	print "shape of feature is ",feature.shape
	data=np.load("database/data.npy")
	labels=np.load("database/labels.npy")
	distance=[]
	for i in data:
		distance.append(scipy.spatial.distance.euclidean(i,feature))
	distance,labels=zip(*sorted(zip(distance,labels)))
	return labels