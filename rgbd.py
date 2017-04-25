import cv2
import numpy as np
from saliency import saliency
from histogram import histogram
from entropy import entro
import matplotlib.pyplot as plt
def rgbd( rgb , depth_map ):
	#extract facial region using viola jones later


	#make size of image ideal
	height,width=rgb.shape[:2]
	#extract patch 1
	h1 = int(0.125*height)
	h2 = int(0.875*height)
	w1 = int(0.125*width)
	w2 = int(0.875*width)
	#patch1=rgb[0.125*height:0.875*height,0.125*width:0.875*width]
	patch1=rgb[h1:h2,w1:w2]	
	patch1=np.array(patch1)
	#print type(patch1)

	#extract patch2
	h1 = int(0.25*height)
	h2 = int(0.75*height)
	w1 = int(0.25*width)
	w2 = int(0.75*width)

	#patch2=rgb[0.25*height:0.75*height,0.25*width:0.75*width]
	patch2=rgb[h1:h2,w1:w2]
	
	h1 = int(0.125*height)
	h2 = int(0.875*height)
	w1 = int(0.125*width)
	w2 = int(0.875*width)	
	
	#patch3 = depth_map[0.125*height:0.875*height,0.125*width:0.875*width]
	patch3=depth_map[h1:h2,w1:w2]

	h1 = int(0.25*height)
	h2 = int(0.75*height)
	w1 = int(0.25*width)
	w2 = int(0.75*width)	

	#patch4 = depth_map[0.25*height:0.75*height,0.25*width:0.75*width]
	patch4=depth_map[h1:h2,w1:w2]	
	#print "original size is",image.shape
	#print "patch1 size is",patch1.shape
	#print "patch2 size is",patch2.shape
	E1 = entro(patch1)
	E2 = entro(patch2)
	S = saliency(rgb)
	E3 = entro(patch3)
	E4 = entro(patch4)
	#return [E1,E2,E3,E4,S]
	final_hog=[]
	for i in [E1,E2,E3,E4,S]:
		i=np.array(i,np.uint8)
		#final_hog=np.concatenate(final_hog,np.histogram(i,bins=255))
		#final_hog.append(histogram(i))
		final_hog.append(np.histogram(i,bins=range(256))[0])
	#print "feature size is ",np.array(final_hog).flatten().shape
	return np.array(final_hog).flatten()
if __name__ == "__main__":
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
	k=1
	for i in final_answer:
		print np.histogram(i,bins=range(256))[0]
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.imshow(final_answer[i],cmap='gray')
	plt.show()
	cv2.imshow("saliency map",final_answer[4])
	cv2.waitKey(0)
	#print final_answer.flatten()
	#print final_answer.shape

