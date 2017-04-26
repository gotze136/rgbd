import cv2
import numpy as np
import os
import easygui
from collections import Counter
from rgbd import rgbd
import scipy
import traceback
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]
success=0
fail=0
total=0
data=np.load("database/data.npy")
naming=np.load("database/naming.npy")
labels=np.load("database/labels.npy")
locations=np.load("database/locations.npy")

path=easygui.diropenbox("select the directory you want to check your database on")
for persons in os.listdir(path):
	goal=persons
	color_path=path+'/'+persons+'/RGB'
	for color_images in os.listdir(color_path):
		color_image=color_path+'/'+color_images
		rgb=cv2.imread(color_image)
		rgb=cv2.resize(rgb,(140,140))
		depth_path=color_image.split("/")[::-1]
		
		depth_path[1]="Depth"
		depth_path=depth_path[::-1]
		depth_path="/".join(depth_path)
		#depth_path=depth_path+'/'+color_images
		#print depth_path
		try:
			depth=cv2.imread(depth_path)
			depth=cv2.resize(depth,(140,140))
		except:
			"depth not read properly for ",color_path
			continue
		feature=rgbd(rgb,depth)
		#print feature
		try:
			distance=[]
			k=0
			for i in data:
				#print i.shape,feature.shape
				dist=scipy.spatial.distance.euclidean(i,feature)
				#print locations[k],labels[k],dist
				#print dist
				distance.append(dist)
				k+=1
			temp=labels.copy()
			distance,temp=zip(*sorted(zip(distance,temp)))
			temp=[naming[i] for i in temp]
			person=Most_Common(temp[:10])
			if person==persons:
				success+=1
			else:
				fail+=1
				print "matched with ",persons,temp[:10]
			total+=1
		except Exception as err:
			try:
				raise TypeError("Again !?!")
			except:
				pass
			traceback.print_tb(err.__traceback__)

		print "checking ",color_image,success,fail,total
		print str(float(success*100)/total)+'%'

