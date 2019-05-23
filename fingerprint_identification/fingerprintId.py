import cv2, os, sys, math
from os import listdir
from os.path import join,isdir,isfile

from PIL import Image
#from matplotlib import pyplot as plt
import numpy as np
import json

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
		self.valid_blocks = []
		self.angle_blocks = []
		self.grad_blocks = []
		self.singularity = []
		self.images = []
		self.images_classes = []
		self.original_classes = []
		self.names = []
		self.load_images()

	def load_images(self):
		files = [f for f in listdir(self.path) if isfile(join(self.path, f)) and f.endswith(".raw")]
		for file in files:
			signs = {'delta': 0, 'core': 0}
			file_path = self.path+'/'+file
			if self.path == 'Rindex28':
				json_path = self.path+'-type/'+file[0:6]+'.lif'
				json_data = json.loads(open(json_path).read())
				for sign in json_data['shapes']:
					# print(sign)
					if sign['label'] == 'core':
						signs['core']+=1
					elif sign['label'] == 'delta':
						signs['delta']+=1
			# print(signs)
			# print(file_path)
			img = Image.frombytes('L',self.size, open(file_path).read(), decoder_name='raw')
			img = np.array(img, 'uint8')
			#cv2.imshow("original",img)
			#cv2.waitKey(0)
			eimg = self.enhacement(img)
			# print(eimg.shape)
			eimg = self.orientation(eimg)
			# print(eimg.shape)
			self.region_interest_detection(img)
			dimg = self.test_detection(img)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			eimg = cv2.cvtColor(eimg,cv2.COLOR_BGR2RGB)
			coordinates, eimg = self.singular_point_detection(eimg)
			class_image = self.classify_fingerprint(coordinates)
			img = self.show_regions(img, coordinates)
			#print(type)
			# cv2.imshow("detection"+file,dimg)
			# cv2.imshow("orientation"+file,dimg)
			# cv2.imshow("detect img"+class_image,img)
			# cv2.waitKey(0)
			self.images.append(img)
			self.images_classes.append(class_image)
			self.original_classes.append(signs)
			print(file[0:6])
			self.names.append(file[0:6])

	def show_regions(self ,img, coordinates):
		delta = coordinates['delta']
		loop = coordinates['loop']
		whorl = coordinates['whorl']
		for i in loop:
			cv2.circle(img,(int(i[0]),int(i[1])),5,(0,255,0),3)			
		for i in delta:
			cv2.circle(img,(int(i[0]),int(i[1])),5,(255,0,0),3)			
		for i in whorl:
			cv2.circle(img,(int(i[0]),int(i[1])),5,(0,0,255),3)
		return img			

	def enhacement(self, img):
		dimg = np.copy(img)
		mean = img.mean()
		variance = img.var()
		s = 96
		y = 95
		alfa = 150
		# print("BEFORE",img)
		while s > y:
			mean = dimg.mean()
			variance = dimg.var()
			s = np.sqrt(variance)
			eimg = []
			for i in dimg:
				row = []
				for j in i:
					p = alfa+y*((j-mean)/s)
					row.append(p)
				eimg.append(row)
		eimg = np.array(eimg)
		eimg = scale_image(eimg)
		# print("AFTER",dimg)
		return eimg

	def orientation(self, img):
		img = np.array(img)
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
		block_size = 15
		block_size_x = int(np.floor(self.size[0] / block_size))
		block_size_y = int(np.floor(self.size[1] / block_size))
		self.block_size = [block_size, block_size_x, block_size_y]
		size = (block_size_x,block_size_y)
		grad_vectors = self.calc_gradient_vectors(grad_img)
		average_blocks = np.empty(size, dtype=object)
		for bi in range(0, block_size_x):
			for bj in range(0, block_size_y):
				bgradients = []
				for i in range(bi * block_size, (bi * block_size) + block_size):
					for j in range(bj * block_size, (bj * block_size) + block_size):
						bgradients.append(grad_vectors[i][j])
				bsum = np.sum(bgradients,axis=0)
				bmean = [bsum[0] / (block_size * block_size), bsum[1] / (block_size * block_size)]
				average_blocks[bi][bj] = bmean
		self.grad_blocks = average_blocks
		#PRINT ORIENTATION BLOCKS(2,4)
		self.angle_blocks = [[ 0 for i in range(0,self.block_size[1])] for j in range(0,self.block_size[2])]
		for i in range(0,block_size_x):
			for j in range(0, block_size_y):
				radians = 0.5 * np.arctan2(average_blocks[i][j][1],average_blocks[i][j][0]) + np.pi/2
				self.angle_blocks[i][j] = radians
				# print(radians,0.5*np.arctan2(average_blocks[i][j][1],average_blocks[i][j][0]))
				inv_radians = radians + np.pi
				center_line = ((j * block_size) + (np.ceil(block_size/2)), i * block_size + np.ceil(block_size/2))
				center_line = (int(center_line[0]),int(center_line[1]))

				end_line0 = (center_line[0] + 6 * np.cos(radians),center_line[1] + 6 * np.sin(radians))
				end_line0 = (int(end_line0[0]),int(end_line0[1]))

				end_line1 = (center_line[0] + 6 * np.cos(inv_radians),center_line[1] + 6 * np.sin(inv_radians))
				end_line1 = (int(end_line1[0]),int(end_line1[1]))
				cv2.line(img, end_line0, end_line1, (0,0,0), 3)
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
		self.valid_blocks = [[ 0 for i in range(0,self.block_size[1])] for j in range(0,self.block_size[2])]
		for bi in range(0, self.block_size[1]):
			for bj in range(0, self.block_size[2]):
				block_pixels = []
				for i in range(bi * self.block_size[0], (bi * self.block_size[0]) + self.block_size[0]):
					for j in range(bj * self.block_size[0], (bj * self.block_size[0]) + self.block_size[0]):
						block_pixels.append(img[i][j])
				bx = bi * self.block_size[0] + self.block_size[0]/2
				by = bj * self.block_size[0] + self.block_size[0]/2
				major_distance_center = (self.size[0])*(math.sqrt(2)/2)
				distance_block_center = math.sqrt( (bx - (self.size[0]/2))**2 + (by - (self.size[1]/2))**2)
				# print("DIST", bx,by,major_distance_center,distance_block_center)
				v = 0.7 * (1 - (np.mean(block_pixels)/max(block_pixels))) + 0.7 * (np.std(block_pixels)/max(block_pixels)) + 0.7*(1-(distance_block_center/major_distance_center))
				# print("DIST",v, 0.5 * (1 - (np.mean(block_pixels)/max(block_pixels))),0.5 * (np.std(block_pixels)/max(block_pixels)),(distance_block_center/major_distance_center))
				# cv2.circle(img,(bx,by),5,(0,255,0))
				# cv2.imshow('test_intes',img)
				# cv2.waitKey(0)

				if(v <= 0.50):
					self.valid_blocks[bi][bj] = 1
		# print(self.valid_blocks)

	def test_detection(self, img):
		tdimg = np.copy(img)
		for bi in range(0, self.block_size[1]):
			for bj in range(0, self.block_size[2]):
				if(self.valid_blocks[bi][bj]):
					for i in range(bi * self.block_size[0], (bi * self.block_size[0]) + self.block_size[0]):
						for j in range(bj * self.block_size[0], (bj * self.block_size[0]) + self.block_size[0]):
							tdimg[i][j] = 0
		return tdimg


	def singular_point_detection(self, img):
		near_blocks = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
		coordinates_loop = [] #coordinates of interest
		coordinates_delta = [] 
		coordinates_whorl = [] 
		interval = 0.5
		for bi in range(1, self.block_size[1]-1):
			for bj in range(1, self.block_size[2]-1):
				if(not self.valid_blocks[bi][bj]) and \
						(not self.valid_blocks[bi-1][bj-1]) and (not self.valid_blocks[bi+1][bj+1]) and \
						(not self.valid_blocks[bi+1][bj]) and (not self.valid_blocks[bi-1][bj]) and \
						(not self.valid_blocks[bi][bj+1]) and (not self.valid_blocks[bi][bj-1]):
					#smoth the direction
					# a = self.grad_blocks[bi][bj][0]*2
					# b = self.grad_blocks[bi][bj][1]*2
					# for i in range(0,8):
					# 	a += self.grad_blocks[bi+near_blocks[i][0]][bj+near_blocks[i][1]][0]
					# 	b += self.grad_blocks[bi+near_blocks[i][0]][bj+near_blocks[i][1]][1]
					# self.angle_blocks[bi][bj] = 0.5*np.arctan(b/a)


					#get poincare
					poincare_index = 0
					previus = near_blocks[0]
					for i in range(1,len(near_blocks)):
						angle_previus = np.degrees(self.angle_blocks[bi+previus[0]][bj+previus[1]])
						angle_atual = np.degrees(self.angle_blocks[bi+near_blocks[i][0]][bj+near_blocks[i][1]])
						if abs(get_angle(angle_previus,angle_atual))>90:
							angle_atual += 180	
						poincare_index +=  get_angle(angle_previus,angle_atual)
						previus = near_blocks[i]
					poincare_index = poincare_index

					coord = (bj * self.block_size[0] + self.block_size[0]/2,bi * self.block_size[0]  + self.block_size[0]/2)
					if (180 - interval <= poincare_index) and (poincare_index <= 180 + interval):
						coordinates_loop.append(coord)
						# print("loop", coord)
						cv2.circle(img,coord,5,(0,255,0),3)
					if (-180 - interval <= poincare_index) and (poincare_index <= -180 + interval):
						# print("delta",coord)
						coordinates_delta.append(coord)
						cv2.circle(img,coord,5,(255,0,0),3)
					if (360 - interval <= poincare_index) and (poincare_index <= 360 + interval):
						# print("whorl",coord)
						coordinates_whorl.append(coord)
						cv2.circle(img,coord,5,(0,0,255),3)
		coordinates_loop = group_coordinates(coordinates_loop,(self.block_size[0]*np.sqrt(2)))
		coordinates_whorl = group_coordinates(coordinates_whorl,(self.block_size[0]*np.sqrt(2)))
		coordinates_delta = group_coordinates(coordinates_delta,(self.block_size[0]*np.sqrt(2)))
		coordinates = {}
		coordinates['loop'] = coordinates_loop
		coordinates['delta'] = coordinates_delta
		coordinates['whorl'] = coordinates_whorl
		return coordinates, img

	def classify_fingerprint(self, coordinates):
		cores, deltas = count_sign(coordinates)
		if(cores == 0 and deltas == 0):
			return('other')
		if(cores == 2 and deltas >= 2):
			return('whorl')
		if(cores == 1 and deltas == 0):
			return('arch')
		if(cores == 0 and deltas == 1):
			return('arch')
		if(cores == 1 and deltas == 1):
			return(define_side(coordinates))
		return('other')



def count_sign(coordinates):
	ncore = len(coordinates['loop'])
	ndelta = len(coordinates['delta'])
	return (ncore, ndelta)

def define_side(coordinates):
	if(coordinates['loop'][0][0] > coordinates['delta'][0][0]):
		return ('right_loop')
	return ('left_loop')

def group_coordinates(coordinates, max_distance):
	if len(coordinates) == 0:
		return []
	groups = {}
	groups[0] = [coordinates[0]]
	g_count = 1
	for i,x in enumerate(coordinates[1:]):
		is_grouped = False
		for g in groups:
			for coord in groups[g]:
				if np.sqrt( (coord[0]-x[0])**2 + (coord[1]-x[1])**2) <= max_distance:
					groups[g].append(x)
					is_grouped = True        
					break
			if is_grouped:
				break
		if not is_grouped:
			groups[g_count] = [x]
			g_count+=1        
	
	return [np.mean(groups[g], axis=0) for g in groups]
	# return [np.median(groups[g], axis=0) for g in groups]


signum = lambda x: -1 if x < 0 else 1

def get_angle(left, right):
    angle = left - right
    if abs(angle) > 180:
        angle = -1 * signum(angle) * (360 - abs(angle))
    return angle


def scale_image(arr):
	# print(arr.min(),arr.max(),arr)
	new_arr = (((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8'))
	return new_arr

def main(argv):
	# lind = FingerprintId(DBPath['lindex'])
	rind = FingerprintId(DBPath['rindex'])
	classes = zip(rind.images_classes,rind.original_classes)
	acc = 0
	err = 0
	all = 0
	for computed,original in classes:
		all=all+1
		# print(computed,original)
		cores = original['core']
		deltas = original['delta']
		# print(cores,deltas)
		if cores == 0 and deltas == 0:
			if 'other' == computed:
				acc=acc+1
			else:
				err=err+1
		elif cores == 2 and deltas >= 2:
			if 'whorl' == computed:
				acc=acc+1
				# print(original,computed)
			else:
				err=err+1
		elif cores == 1 and deltas == 0:
			if 'arch' == computed:
				acc=acc+1
				# print(original,computed)
			else:
				err=err+1
		elif cores == 0 and deltas == 1:
			if 'arch' == computed:
				acc=acc+1
				# print(original,computed)
			else:
				err=err+1
		elif cores == 1 and deltas == 1:
			if computed == 'right_loop' or computed == 'left_loop':
				acc=acc+1
				print(original,computed)
			else:
				err=err+1
		else:
			err=err+1
	print("todos",all)
	print("erros",err)
	print("acertos",acc)
	print("acuracia",(acc*1.0)/all)
if __name__ == "__main__":
	main(sys.argv[1:])
