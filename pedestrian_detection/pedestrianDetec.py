import cv2, os, sys, math
from os import listdir
from os.path import join,isdir,isfile

from PIL import Image
#from matplotlib import pyplot as plt
import numpy as np
import json

waitingtime = 0.1

DBPath = {
	"train": "Train",
	"test": "Test"
}

# image_path = '/home/html/inf/menotti/ci1028-191/INRIAPerson/'

class Pedestrian:

	def __init__(self, path):
		self.path = 'INRIAPerson/'+path
		self.annotations = []
		self.load_images()

	def load_images(self):
		limit = 10
		text_annotations = open(self.path+'/annotations.lst','r')
		for index,text in enumerate(text_annotations):
			annotations = open('INRIAPerson/'+text[0:-1])
			print(annotations)
			if index == limit:
				break
		return None



def main(argv):
	rind = Pedestrian(DBPath['train'])


if __name__ == "__main__":
	main(sys.argv[1:])