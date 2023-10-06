import cv2
from utils import process_frame
#from skimage.io import imsave

# try to read input as image
image = cv2.imread("./data/demo2.png")

if image is not None:
	#success, it was an image
	viz = process_frame(image)
	cv2.imshow('', viz)
	cv2.waitKey(0)
	#imsave('./result2.png', viz)