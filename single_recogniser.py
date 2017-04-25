import cv2
import easygui
import numpy as np
from euclidean_matcher import matcher
color_path=easygui.fileopenbox("choose color image")

rgb=cv2.imread(color_path)
rgb=cv2.resize(rgb,(140,140))
depth_path=color_path.split("/")[::-1]
depth_path[1]="Depth"
depth_path=depth_path[::-1]
depth_path="/".join(depth_path)
print depth_path
depth=cv2.imread(depth_path)
depth=cv2.resize(depth,(140,140))
labels=matcher(rgb,depth)[:10]
naming=np.load("database/naming.npy")
for i in labels:
	print naming[i],
#print labels[:10]

