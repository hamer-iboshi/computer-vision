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
		self.block_size = []
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
			img = np.array(img)
			cv2.imshow(file,img)
			cv2.waitKey(0)
			eimg = self.enhacement(img)
			print(eimg.shape)
			eimg = self.orientation(eimg)
			print(eimg.shape)
			cv2.imshow(file+'2',eimg)
			cv2.waitKey(0)
			self.images.append(eimg)
			print(file[0:4])
			self.images_classes.append(file[0:4])

	def enhacement(self, img):
		dimg = np.copy(img)
		mean = img.mean()
		variance = img.var()
		s = 96
		y = 95
		alfa = 150
		print("BEFORE",img)
		while s > y:
			mean = dimg.mean()
			variance = dimg.var()
			s = np.sqrt(variance)
			eimg = []
			print("S < Y ",s)
			for i in dimg:
				row = []
				for j in i:
					p = alfa+y*((j-mean)/s)
					row.append(p)
				eimg.append(row)
		eimg = np.array(eimg)
		eimg = scale_image(eimg)
		print("AFTER",dimg)
		return eimg

	def orientation(self, img):
		#SOBEL (1)
		blur_img = np.copy(img)
		blur_img = cv2.blur(blur_img, (5, 5))
		sobelx = cv2.Sobel(blur_img, cv2.CV_64F, 1, 0, ksize=5)
		sobely = cv2.Sobel(blur_img, cv2.CV_64F, 0, 1, ksize=5)
		grad_img = np.empty(self.size, dtype=object)
		for i in range(0,self.size[0]):
			for j in range(0, self.size[1]):
				grad_img[i][j] = [sobelx[i][j], sobely[i][j]]

		#AVERAGE ORIENTATION (2,3)
		block_size = 10
		block_size_x = int(np.floor(self.size[0] / block_size))
		block_size_y = int(np.floor(self.size[1] / block_size))
		self.block_size = [block_size, block_size_x, block_size_y]
		size = (block_size_x,block_size_y)
		grad_vectors = self.calc_gradient_vectors(grad_img)
		average_blocks = np.empty(size, dtype=object)
		for bi in range(0, block_size_x):
			for by in range(0, block_size_y):
				bgradients = []
				for i in range(bi * block_size, (bi * block_size) + block_size):
					for j in range(by * block_size, (by * block_size) + block_size):
						bgradients.append(grad_vectors[i][j])
				bsum = np.sum(bgradients,axis=0)
				bmean = [bsum[0] / (block_size * block_size), bsum[1] / (block_size * block_size)]
				average_blocks[bi][by] = bmean

		#PRINT ORIENTATION BLOCKS(2,4)
		for i in range(0,block_size_x):
			for j in range(0, block_size_y):
				radians = 0.5 * np.arctan2(average_blocks[i][j][1],average_blocks[i][j][0]) + np.pi/2
				# print(radians,0.5*np.arctan2(average_blocks[i][j][1],average_blocks[i][j][0]))
				inv_radians = radians + np.pi
				center_line = ((j * block_size) + (np.ceil(block_size/2)), i * block_size + np.ceil(block_size/2))
				center_line = (int(center_line[0]),int(center_line[1]))

				end_line0 = (center_line[0] + 6 * np.cos(radians),center_line[1] + 6 * np.sin(radians))
				end_line0 = (int(end_line0[0]),int(end_line0[1]))

				end_line1 = (center_line[0] + 6 * np.cos(inv_radians),center_line[1] + 6 * np.sin(inv_radians))
				end_line1 = (int(end_line1[0]),int(end_line1[1]))
				cv2.line(img, end_line0, end_line1, (0,0,0), 1)
		return img

	def calc_gradient_vectors(self, gradient):
		calc_gradient = np.empty(self.size, dtype=object)
		for i in range(0,self.size[0]):
			for j in range(0, self.size[1]):
				g = gradient[i][j]
				calc_gradient[i][j] = [(g[0] * g[0] - g[1] * g[1]),(2 * g[0] * g[1])]
		return calc_gradient
	
	def region_interest_detection(self, img):
	#calculate mean and standard deviation from each block for shades of grey
		for bi in range(0, self.block_size[1]):
			for bj range(0, self.block_size[2]):
				block_pixels = []
				for i in range(bi * self.block_size[1], (bi * self.block_size[1]) + self.block_size[1]):
					for j in range(bj * self.block_size[2], (bj * self.block_size[2]) + self.block_size[2]):
						block_pixels.append(img[i][j])
				ratio_d = 
				v = 0.5 * (1 - np.mean(block_pixels)) + 0.5 * np.std(block_pixels) + ratio_d
				if( v > 0.8):
	
def scale_image(arr):
	print(arr.min(),arr.max(),arr)
	new_arr = (((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8'))
	return new_arr

def main(argv):
	fp = FingerprintId(DBPath['lindex'])


if __name__ == "__main__":
	main(sys.argv[1:])
