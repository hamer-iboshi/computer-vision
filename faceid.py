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
		return mean/len(self.images)

	def eigenFaces(self, Ml):
		mface = self.meanFace()
		meanFace = np.reshape(mface,(-1))
		x = []
		for img in self.images:
			x.append(np.reshape(img,(-1)))
			#print(img.shape,x)
		#print(x)
		x = np.array(x)
		#print(x,x.shape,meanFace,meanFace.shape)
		r = np.subtract(x,meanFace)
		c = np.dot(r,r.T)
		w,v = np.linalg.eig(c) #eigen values, eigen vectors
		eigenFace = []
		print(r.shape,v.shape)
		for Ai in r:
			eigenFaces = np.dot(Ai,v[0])
			print(eigenFaces.shape)
	
		return eigenFaces	
	
	def eigenFaces2Img(self,efaces):

		#new_arr = ((ri - ri.min()) * (1/(ri.max() - ri.min())) * 255).astype('uint8')
		pass

class ORLFaces(FaceId):
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
			
class YaleFaces(FaceId):
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
