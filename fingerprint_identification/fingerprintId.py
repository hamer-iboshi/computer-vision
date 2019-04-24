import cv2, os, sys
from os import listdir
from os.path import join,isdir,isfile

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

waitingtime = 0.1
	
DBPath = {
	"lindex": "Lindex101",
	"rindex": "Rindex28"
}

class FingerprintId:

	def __init__(self, path,size = (300,300)):
		self.path = path
		self.size = size
		self.images = []
		self.images_classes = []
		self.load_images()

	def load_images(self):
		files = [f for f in listdir(self.path) if isfile(join(self.path, f)) and f.endswith(".raw")]
		for file in files:
			file_path = self.path+'/'+file
			print(file_path)
			img = Image.frombytes('L',self.size, open(file_path).read(), decoder_name='raw')
			img.load()
			cimg = np.asarray(img)
			#cv2.imshow(file,img)
			#cv2.waitKey(0)
			eimg = self.enhacement(cimg)
			cv2.imshow(file,eimg)
			cv2.waitKey(0)
			self.images.append(eimg)
			print(file[0:4])
			self.images_classes.append(file[0:4])

	def enhacement(self, img):
		dimg = np.copy(img)
		size_i = self.size[0]
		size_j = self.size[1]
		mean = img.mean()
		variance = img.var()
		s = 96
		y = 95
		alfa = 150 
		while s > y:
			mean = dimg.mean()
			variance = dimg.var()
			s = np.sqrt(variance)
			print("S < Y ",s)
			for i in range(0,size_i):
				for j in range(0,size_j):
					dimg[j][i]=alfa+y*((img[i][j]-mean)/s)
			
		print(dimg)
		dimg = scale_image(dimg)
		return dimg

def scale_image(arr):
	new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min())) * 255).astype('uint8')
	return new_arr

def main(argv):
	fp = FingerprintId(DBPath['lindex'])


if __name__ == "__main__":
	main(sys.argv[1:])
