import cv2, os
from os import listdir
from os.path import isfile, isdir, join

from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


class FaceId:
	DBPath = {
		"yale": "yale_faces",
		"orl": "orl_faces",
	}

	def __init__(self,path,_bFaceDetect,size = (100,100)):
		self.path = path
		self.bFaceDetect = _bFaceDetect
		self.size = size
		self.images = []
		self.images_classes = []
		self.load_images()
		#plt.imshow(self.meanFace(), 'gray')
		#plt.show()
		
	def meanFace(self):
		mean = np.zeros(self.size, dtype=np.float64)
		for img in self.images:
			mean+=img
			print(img.shape)
		return mean/len(self.images)

	def eigenFaces(self, Ml):
		mface = self.meanFace()
		x1 = np.reshape(mface,(-1))
		print(x1,x1.shape)
		
class ORLFaces(Database):
	def load_images(self):
		classes = [f for f in listdir(self.path) if isdir(join(self.path, f))]
		
		for class_name in classes:
			class_path = self.path+"/"+class_name
			files = [f for f in listdir(class_path) if isfile(join(class_path, f))]

			for file in files:
				file_path = self.path+"/"+class_name+"/"+file
				
				img = Image.open(file_path)
				img.load()
				img = np.asarray(img)
				img = cv2.resize(img,self.size)
				class_file = class_name
				
				self.images.append(img)
				self.images_classes.append(class_file)
			
class YaleFaces(Database):
	def load_images(self):
		files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
		for file in files:
			file_path = self.path+"/"+file
			
			img = Image.open(file_path)
			img.load()
			img = np.asarray(img)
			
			x, y, w, h = faceCascade.detectMultiScale(img)[0]
			img = cv2.resize(img[y: y + h, x: x + w], self.size)
			
			class_file = file.split(".")[0]
			
			self.images.append(img)
			self.images_classes.append(class_file)

## Path to the Yale Dataset
#path = '/home/menotti/databases/yalefaces/'
#print 'loading Yalefaces database'
#yale = YaleFaces(path)
#yale.eigenFaces2()

## Path to the ORl Dataset
#path = '/home/menotti/databases/orl_faces/'
#print 'loading ORL database'
#orl = ORL(path)
#orl.eigenFaces2()
