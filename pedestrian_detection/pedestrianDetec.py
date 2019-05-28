import cv2, os, sys, math
from os import listdir
from os.path import join,isdir,isfile

from PIL import Image
#from matplotlib import pyplot as plt
import numpy as np
import imutils

waitingtime = 0.1

DBPath = {
	"train": "Train",
	"test": "Test"
}

image_path = '/home/html/inf/menotti/ci1028-191'

class Pedestrian:

	def __init__(self, path):
		self.path = image_path+'/INRIAPerson/'+path
		self.annotations = []
		self.load_images()

	def load_images(self):
		limit = 4
		text_annotations = open(self.path+'/annotations.lst','r')
		for index,text in enumerate(text_annotations):
			annotations = open(image_path+'/INRIAPerson/'+text[0:-1])
			image_name = text.split('/',-1)[-1][0:-5]
			img = cv2.imread(self.path+'/pos/'+image_name+'.png')
			aimg = np.array(img)
			print(self.path+'/pos/'+image_name,aimg.shape)
			#cv2.imshow('img',img)
			#cv2.waitKey(0)
			if index == limit:
				break
		return None

#calcular a orientacao e a magnitude para cada canal e filtrar pela maior magnitude
def hog_feature_extraction():
	return None

def pyramid_method(img):
	# METHOD #1: No smooth, just scaling.
	# loop over the image pyramid
	for (i, resized) in enumerate(pyramid(img)):
		# show the resized image
		cv2.imshow("Layer {}".format(i + 1), resized)
		cv2.waitKey(0)
	
 
def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image


def main(argv):
	rind = Pedestrian(DBPath['train'])


if __name__ == "__main__":
	main(sys.argv[1:])
