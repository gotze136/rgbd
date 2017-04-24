import os
import sys
import numpy as np
import cv2
from rgbd import rgbd
import easygui
#path=sys.argv[1]
path=easygui.diropenbox("select the root of database")
naming=[]
labels=[]
locations=[]
data=[]
k=0
for folders in os.listdir(path):  #loop over all the persons
	folder=path+'/'+folders
	naming.append(folders)
	depth_dir=folder+'/Depth'
	color_dir=folder+'/RGB'
	for images in os.listdir(color_dir):
		color_image=color_dir+'/'+images
		depth_image=depth_dir+'/'+images
		print "checking for ",color_image
		color=cv2.imread(color_image)
		color=cv2.resize(color,(140,140))
		try:
			depth=cv2.imread(depth_image)
			depth=cv2.resize(depth,(140,140))
		except:
			print "depth image not found for this rgb image ",color_image
			continue
		try:
		
			feature=rgbd(color,depth)
			data.append(feature)
			#print image
			locations.append(color_image)
			labels.append(k)
		except:
			print "failed for ",color_image
		#data.append(hist(calcgrad(cv2.resize(cv2.imread(image,0),(240,240)))))
	k+=1
os.chdir("database")
np.save("naming",np.array(naming))
np.save("labels",np.array(labels))
np.save("locations",np.array(locations))
np.save("data",np.array(data))
