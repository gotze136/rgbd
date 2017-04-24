import cv2
import easygui
from euclidean_matcher import matcher
color_path=easygui.fileopenbox("choose color image")
rgb=cv2.imread(color_path)
rgb=cv2.resize(rgb,(140,140))
depth_path=easygui.fileopenbox("choose depth map path")
depth=cv2.imread(depth_path)
depth=cv2.resize(depth,(140,140))
labels=matcher(rgb,depth)
print labels[:10]

