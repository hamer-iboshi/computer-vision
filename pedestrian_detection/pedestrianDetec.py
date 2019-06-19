import cv2, os, sys, math
from os import listdir
from os.path import join,isdir,isfile
from skimage import feature
from PIL import Image

from sklearn import svm
#from matplotlib import pyplot as plt
import numpy as np
#import imutils
import re

waitingtime = 0.1

DBPath = {
	"train": "Train",
	"test": "Test"
}

# image_path = '/home/html/inf/menotti/ci1028-191'
image_path = '/home/hi15/BCC/BCC9/computer-vision/pedestrian_detection'

class Pedestrian:

	def __init__(self, path):
		self.path = image_path+'/INRIAPerson/'+path
		self.annotations = []
		self.images = []
		self.hogs = []
		self.labels = []
		self.load_images()

	def load_images(self):
		limit = 1
		text_annotations = open(self.path+'/annotations.lst','r')
		for index,text in enumerate(text_annotations):
			annotations = open(image_path+'/INRIAPerson/'+text[0:-1],'r')
			image_name = text.split('/',-1)[-1][0:-5]
			img = cv2.imread(self.path+'/pos/'+image_name+'.png')
			aimg = np.array(img)
			coord = get_persons_coord(annotations)
			#get only first but in the end get all persons
			first = coord[0]
			print(first,aimg.shape,image_name,text)
			#person_img = [[[ aimg[i+first[1]-1][j+first[0]-1][k] for i in range(0,first[3]-first[1])] for j in range(0,first[2]-first[0])] for k in range(0,3) ] 
			person_img = [[[ aimg[i+first[1]][j+first[0]][k] for k in range(0,3)] for j in range(0,first[2]-first[0])] for i in range(0,first[3]-first[1])] 
			person_img = np.array(person_img)
			print(self.path+'/pos/'+image_name,aimg.shape,person_img.shape)
			(H, hogImage) = feature.hog(person_img.copy(), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True,  block_norm="L1",visualize=True)
			self.images.append([person_img,H,'pos'])
			self.hogs.append(H)
			self.labels.append('pos')
			cv2.imshow('img',hogImage)
			cv2.waitKey(0)
			# cv2.imshow('img',person_img)
			# cv2.waitKey(0)
			if index == limit:
				break
		neg_path = self.path+'/neg/'
		files = [f for f in listdir(neg_path) if isfile(join(neg_path, f))]
		for index,file in enumerate(files):
			print(index,file)
			neg_img = cv2.imread(neg_path+file)
			(H, hogImage) = feature.hog(neg_img.copy(), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True,  block_norm="L1",visualize=True)
			self.hogs.append(H)
			self.labels.append('neg')
			cv2.imshow('img',hogImage)
			cv2.waitKey(0)
			if index == limit:
				break

#calcular a orientacao e a magnitude para cada canal e filtrar pela maior magnitude
def hog_feature_extraction(img):
	
	cv2.imshow('img',aimg)
	cv2.wait(0)
	# return None

def get_persons_coord(annotation):
	coord = []
	regex = r"\s\([\d]+,\s[\d]+\)\s\-\s\([\d]+\,\s[\d]+\)"
	text_coords = re.findall(regex, annotation.read(), re.MULTILINE)
	for text_coord	in text_coords:
		value_coord = re.findall(r"[\d]+",text_coord, re.MULTILINE)
		value_coord = map(int,value_coord)
		coord.append(value_coord)
	return coord

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
	S = svm(rind.labels,rind.hogs)

if __name__ == "__main__":
	main(sys.argv[1:])
