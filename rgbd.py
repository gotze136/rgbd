import cv2
import numpy as np
from saliency import saliency
from histogram import histogram
from entropy import entro
def rgbd( rgb , depth_map ):
	#extract facial region using viola jones later


	#make size of image ideal
	height,width=rgb.shape[:2]
	#extract patch 1
	patch1=rgb[0.125*height:0.875*height,0.125*width:0.875*width]
	patch1=np.array(patch1)
	#print type(patch1)

	#extract patch2

	patch2=rgb[0.25*height:0.75*height,0.25*width:0.75*width]
	patch3 = depth_map[0.125*height:0.875*height,0.125*width:0.875*width]
	patch4 = depth_map[0.25*height:0.75*height,0.25*width:0.75*width]
	#print "original size is",image.shape
	#print "patch1 size is",patch1.shape
	#print "patch2 size is",patch2.shape
	E1 = entro(patch1)
	E2 = entro(patch2)
	S = saliency(rgb)
	E3 = entro(patch3)
	E4 = entro(patch4)
	final_hog=[]
	for i in [E1,E2,E3,E4,S]:
		i=np.array(i,np.uint8)
		final_hog.append(histogram(i))
	return final_hog
if __name__ == __main__:
	rgb=cv2.imread("images/color.jpg")
	if rgb is None:
		print "color image is not read properly"
		exit()

	depth=cv2.imread("images/depth.jpg")
	if depth is None:
		print "depth image is not read properly"
		exit()
	#print depth
	final_answer=rgbd(rgb,depth)
	for i in final_answer:
		print np.array(i).shape

