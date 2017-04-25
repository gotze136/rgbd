import cv2
from rgbd import rgbd
import numpy as np
import scipy
def matcher(rgb,depth):
	feature=rgbd(rgb,depth)
	print "shape of feature is ",feature.shape
	data=np.load("database/data.npy")
	labels=np.load("database/labels.npy")
	locations=np.load("database/locations.npy")
	#naming=np.load("database/naming.npy")
	distance=[]
	k=0
	for i in data:
		dist=scipy.spatial.distance.euclidean(i,feature)
		#print locations[k],labels[k],dist
		distance.append(dist)
		k+=1

	distance,labels,locations=zip(*sorted(zip(distance,labels,locations)))
	return labels