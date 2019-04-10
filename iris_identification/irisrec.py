#!/usr/bin/python

# Import the required modules
import cv2, sys, os
import math as mt
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy.linalg as npla
import string

# ToDo's
## Retrieving information from mask
## Reading Masking and Normalization
## Feature generation (LBP \& Wavelet)
## Evaluation FAR vs FRR

_waitingtime = 2.0 #0.5

path_env = os.getcwd()

class IrisRec:

    DBEyePath  = dict(
        CASIAIris = {
            'v4-Lamp100'  : path_env + '/databases/CASIA-IrisV4-Lamp-100/',
            'v4-Interval' : path_env + '/databases/CASIA-IrisV4-Interval/'},
        UBIRIS = {
            'v2-40'       : path_env + '/databases/UBIRIS.v2-40/',
            'v2-150'      : path_env + '/databases/UBIRIS.v2-150'} )

    DBMaskPath  = dict(
        CASIAIris = {
            'v4-Lamp100'  : path_env + '/databases/CASIA-IrisV4-Lamp-100-mask/',
            'v4-Interval' : path_env + '/databases/CASIA-IrisV4-Interval-mask/'},
        UBIRIS = {
            'v2-40'       : path_env + '/databases/UBIRIS.v2-40-mask/',
            'v2-150'      : path_env + '/databases/UBIRIS.v2-150-mask/'} )

    DBIrisPath  = dict(
        CASIAIris = {
            'v4-Lamp100'  : path_env + '/databases/CASIA-IrisV4-Lamp-100-iris/',
            'v4-Interval' : path_env + '/databases/CASIA-IrisV4-Interval-iris/'},
        UBIRIS = {
            'v2-40'       : path_env + '/databases/UBIRIS.v2-40-iris/',
            'v2-150'      : path_env + '/databases/UBIRIS.v2-150-iris/'} )

    DBNormPath  = dict(
        CASIAIris = {
            'v4-Lamp100'  : path_env + '/databases/CASIA-IrisV4-Lamp-100-norm/',
            'v4-Interval' : path_env + '/databases/CASIA-IrisV4-Interval-norm/'},
        UBIRIS = {
            'v2-40'       : path_env + '/databases/UBIRIS.v2-40-norm/',
            'v2-150'      : path_env + '/databases/UBIRIS.v2-150-norm/'} )



    _pathEye  = '' ## virtual path
    _pathMask = ''
    _pathIris = ''
    _pathNorm = ''


    def __init__(self, pathEye = _pathEye, pathMask = _pathMask, pathIris = _pathIris, pathNorm = _pathNorm):
        self.pathEye  = pathEye
        self.pathMask = pathMask
        self.pathIris = pathIris
        self.pathNorm = pathNorm
        if self.pathEye:
            self.getEyeImages(self.pathEye,self.pathMask,self.pathIris,self.pathNorm)
    # elif self.Mask:
    # 	self.getMaskImages(self.pathEye,self.pathMask,self.pathIris,self.pathNorm)
    # elif self.pathNorm:
    # 	self.getIrisImages(self.pathNorm)



# 	def Mask2Norm(self, imgIris, imgMask, wNorm):
# 		# angle points to normalize iris
# 		pts_norm = np.transpose([(mt.cos((2*mt.pi/wNorm[1])*ang), mt.sin((2*mt.pi/wNorm[1])*ang)) for ang in xrange(wNorm[1])])

# 		se3R = cv2.getStructuringElement(cv2.MORPH_RECT   ,(5,5)) # se 3x3 - Squar-shaped
# 		cy,cx = np.divide(imgIris.shape,2) ## extract info from imgMask
# 		radPupil  = 20
# 		radIris   = 40

# #		imgIris = np.zeros(wNorm)
# 		norm_rad = []
# 		for i in range (wNorm[0]):
# 			norm_rad.append((int)(i * ((float)(radIris - radPupil)/wNorm[0]))+0.5+radPupil)

# 		imgNorm = np.zeros((wNorm))

# 		for j in range(wNorm[1]):
# 			for i in range(wNorm[0]):
# 				pt  = ( cy+pts_norm[1][j]*(norm_rad[i]) , cx+pts_norm[0][j]*(norm_rad[i]) )
# 				ptl = ( int(mt.floor(pt[0])), int(mt.floor(pt[1])) )

# 				pt1 = ( ptl[0]  , ptl[1]  , imgIris[ptl[0]  , ptl[1]  ] )
# 				pt2 = ( ptl[0]+1, ptl[1]  , imgIris[ptl[0]+1, ptl[1]  ] )
# 				pt3 = ( ptl[0]  , ptl[1]+1, imgIris[ptl[0]  , ptl[1]+1] )
# 				pt4 = ( ptl[0]+1, ptl[1]+1, imgIris[ptl[0]+1, ptl[1]+1] )

# 				# interpolate
# 				imgNorm[i][j] = bilinear_interpolation(pt[0], pt[1],[pt1,pt2,pt3,pt4])

# 		return imgNorm



class CASIAIris(IrisRec):

    _pathEye  = './CASIA-IrisV4-Interval'
    _pathEye  = './CASIA-IrisV4-Lamp-100'
    _pathMask = './CASIA-IrisV4-Lamp-100-mask'
    _pathIris = './CASIA-IrisV4-Lamp-100-iris'
    _pathNorm = './CASIA-IrisV4-Lamp-100-norm'

    def getEyeImages(self,pathEye = _pathEye, pathMask = _pathMask, pathIris = _pathIris, pathNorm = _pathNorm):
        # images will contains face images
        self.EyeImages  = []
        self.MaskImages = []
        self.IrisImages = []
        self.NormImages = []

        # subjets will contains the subject identification number assigned to the image
        self.idEye = []

        if pathMask and not os.path.exists(pathMask):
            os.makedirs(pathMask)
        if pathIris and not os.path.exists(pathIris):
            os.makedirs(pathIris)
        if pathNorm and not os.path.exists(pathNorm):
            os.makedirs(pathNorm)

        subjects_paths = [os.path.join(pathEye, d) for d in os.listdir(pathEye) if os.path.isdir(os.path.join(pathEye,d))]
        for s,subject_paths in enumerate(subjects_paths, start=1):
            # Get the label of the subject
            print(os.path.split(subject_paths))
            nsb = int(os.path.split(subject_paths)[1])

            if pathMask and not os.path.exists(os.path.join(pathMask,os.path.split(subject_paths)[1])):
                os.makedirs(os.path.join(pathMask,os.path.split(subject_paths)[1]))
            if pathIris and not os.path.exists(os.path.join(pathIris,os.path.split(subject_paths)[1])):
                os.makedirs(os.path.join(pathIris,os.path.split(subject_paths)[1]))
            if pathNorm and not os.path.exists(os.path.join(pathNorm,os.path.split(subject_paths)[1])):
                os.makedirs(os.path.join(pathNorm,os.path.split(subject_paths)[1]))

            side_paths = [os.path.join(subject_paths, d) for d in os.listdir(subject_paths) if os.path.isdir(os.path.join(subject_paths,d))]
            for e,side_path in enumerate(side_paths, start=1):
                idEye = 2*nsb + (-1 if os.path.split(side_path)[1]  == 'L' else 0 )
                print '{0}/{1}:{2}'.format(nsb,idEye,side_path)

                if pathMask and not os.path.exists(os.path.join(pathMask,os.path.split(subject_paths)[1],os.path.split(side_path)[1])):
                    os.makedirs(os.path.join(pathMask,os.path.split(subject_paths)[1],os.path.split(side_path)[1]))
                if pathIris and not os.path.exists(os.path.join(pathIris,os.path.split(subject_paths)[1],os.path.split(side_path)[1])):
                    os.makedirs(os.path.join(pathIris,os.path.split(subject_paths)[1],os.path.split(side_path)[1]))
                if pathNorm and not os.path.exists(os.path.join(pathNorm,os.path.split(subject_paths)[1],os.path.split(side_path)[1])):
                    os.makedirs(os.path.join(pathNorm,os.path.split(subject_paths)[1],os.path.split(side_path)[1]))

                eye_paths = [os.path.join(side_path, f) for f in os.listdir(side_path) if f.endswith('.jpg') and os.path.isfile(os.path.join(side_path,f)) ]
                print("ID",idEye,e,nsb)
                # if(nsb == 80):
                for y, eye_path in enumerate(eye_paths, start=1):
                    # Read the image and convert to grayscale
                    imgEye_pil = Image.open(eye_path).convert('L')
                    # Convert the image format into numpy array
                    imgEye = np.array(imgEye_pil, 'uint8')  # normalization
                    self.EyeImages.append(imgEye)
                    self.idEye.append(idEye)

                    print '{0}:{1}'.format(y, eye_path)
                    sys.stdout.flush()

                    # imgMask, imgIris = self.SegIris(imgEye)
                    imgIris = self.SegIris(imgEye)
                    # if pathMask:
                    # 	imgpathMask = os.path.join(pathMask,os.path.split(subject_paths)[1],os.path.split(side_path)[1],os.path.split(eye_path)[1])
                    # 	cv2.imwrite(imgpathMask,imgMask)
                    if pathIris:
                        imgpathIris = os.path.join(pathIris, os.path.split(subject_paths)[1],
                                                   os.path.split(side_path)[1], os.path.split(eye_path)[1])
                        cv2.imwrite(imgpathIris, imgIris)

                    # imgNorm = self.Mask2Norm(imgIris,imgMask,(32,256))
                    # if pathNorm:
                    # 	imgpathNorm = os.path.join(pathNorm,os.path.split(subject_paths)[1],os.path.split(side_path)[1],os.path.split(eye_path)[1])
                    # 	cv2.imwrite(imgpathNorm,imgNorm)
                    print("Show results.")
                    fig, aplt = plt.subplots(2, 1)
                    aplt[0].imshow(imgEye, cmap='Greys_r')
                    # aplt[0,1].imshow(imgMask,cmap='Greys_r')
                    aplt[1].imshow(imgIris, cmap='Greys_r')
                    # aplt[1,1].imshow(imgNorm,cmap='Greys_r')
                    # plt.savefig('img_test/'+eye_path.split('/')[-1])
                    plt.pause(_waitingtime)
                    plt.close()
                    # exit(0)
                    self.IrisImages.append(imgIris)
                # self.MaskImages.append(imgMask)
                # self.NormImages.append(imgNorm)
                print ' done.'


    def getIrisImages(self,pathEye = _pathEye):
        self.IrisImages = []

    def SegIris(self, imgEye):
        # angle points to find iris border

        ## pupil detection
        print(imgEye.shape)
        # opening - darkening
        imgEye = cv2.cvtColor(imgEye,cv2.COLOR_GRAY2BGR)
        imgBlur = cv2.medianBlur(imgEye,5)
        ret,thresh1 = cv2.threshold(imgBlur,25,255,cv2.THRESH_BINARY_INV)
        # closing
        edges = cv2.Canny(thresh1,0,255)


        #Hough circles transform
        # cimg = imgBlur.copy()
        imgEye = cv2.cvtColor(imgEye,cv2.COLOR_BGR2GRAY)
        cimg = cv2.equalizeHist(imgEye)
        cimg = cv2.cvtColor(cimg,cv2.COLOR_GRAY2BGR)
        #Get circles for pupil
        pupil_min_size = 30
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=62, param2=20, minRadius=pupil_min_size, maxRadius=75)
        while circles is None or self.test_circles(circles) and pupil_min_size>10:
            print("PUPIL_SIZE",pupil_min_size)
            pupil_min_size -= 1
            circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,
                                       param1=62,param2=pupil_min_size,minRadius=pupil_min_size,maxRadius=75)
        circles = np.uint16(np.around(circles))
        print(circles,imgBlur.shape)
        min_med = 600
        min_circle = []
        for i in circles[0,:]:
            med = (i[0]+i[1])/2
            # print("CENTER",i[0],i[1])
            if(med<min_med) and (i[0]>100 and i[0] < 505) and (i[1]>100 and i[1] < 400):
                min_med = med
                min_circle = i
            # draw the outer circle
        print("MIN_CIRCLE",min_circle)
        cv2.circle(cimg,(min_circle[0],min_circle[1]),min_circle[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(min_circle[0],min_circle[1]),2,(0,0,255),3)
        # plt.imshow(cimg)
        # # plt.savefig('img_test/S2036R05.jpg')
        # plt.pause(1)
        # plt.close()
        # computing concentric circles
        conc_circles = []
        intensity = []
        for i in range(0,6):
            circle = [min_circle[0],min_circle[1],min_circle[2]+45+i*2]
            # calc intensity
            circle.append(self.averageCircle(circle,cimg))
            if circle[3] != -1:
                conc_circles.append(circle)
                intensity.append(abs(conc_circles[i-1][3]-circle[3]))
                print("average", conc_circles[i][3])
            else:
                intensity.append(-1)
                break

        max_intensity = np.argsort(intensity)[-1]
        # print(max_intensity,conc_circles,circle[3])
        cv2.circle(cimg,(conc_circles[max_intensity][0],
                        conc_circles[max_intensity][1]),conc_circles[max_intensity][2],
                        (255,0,0),2)

        #try to compute cocentric circles
        # first_average_circle = 10
        # average_circle = 0
        # while first_average_circle > average_circle:

        # fig, aplt = plt.subplots(1,3)
        # aplt[2].imshow(imgBlur,cmap='Greys_r')
        # aplt[1].imshow(cimg,cmap='Greys_r')
        # aplt[0].imshow(edges,cmap='Greys_r')
        # aplt[1,0].imshow(imgIris,cmap='Greys_r')
        # aplt[1,1].imshow(imgNorm,cmap='Greys_r')
        # plt.pause(5)
        # plt.close()
        ## iris contour detection
        # return imgMask,imgIris
        return cimg

    def averageCircle(self, circle, img):
        # print(circle)
        points = 360
        radian = (2 * np.pi) / points
        average = 0
        for i in range(points):
            theta = i * radian
            px = circle[0]+int(circle[2]*np.sin(theta))
            py = circle[1]+int(circle[2]*np.cos(theta))
            # print(px,py,circle)
            if(px >= img.shape[1] or px <= 0) or (py >= img.shape[0] or py <= 0):
                return -1
            # print("IMG P:",img[px][py][0],px,py)
            average += img[py][px][0]
        return average/points

    def test_circles(self, circles):
        for i in circles[0, :]:
            if (i[0]>100 and i[0] < 505) and (i[1]>100 and i[1] < 400):
                return 0
        return 1

class UBIRIS(IrisRec):

    _pathEye  = './UBIRIS.v2-40'
    _pathMask = './UBIRIS.v2-40'

    def getEyeImages(self,pathEye = _pathEye, pathMask = _pathMask):
        # images will contains face images
        self.EyeImages = []
        if pathMask:
            self.MaskImages = []
        # subjets will contains the subject identification number assigned to the image
        self.idEye = []

        imgEye_paths = [os.path.join(pathEye, f) for f in os.listdir(pathEye) if f.startswith('C') and f.endswith('.tiff') and os.path.isfile(os.path.join(pathEye,f))]

        for i,imgEye_path in enumerate(imgEye_paths, start=1):
            # Parse image name to get idEye,idSession,idImg
            imgEye_name = os.path.split(imgEye_path)[1].split('.')[0].translate(string.maketrans("CSI","   "))
            (idEye,idSes,idImg) = [int(id) for id in imgEye_name.split('_')]

            # Read and the image and convert to grayscale and into numpy array
            image_pil = Image.open(imgEye_path).convert('L')
            image = np.array(image_pil, 'uint8') # normalization
            self.EyeImages.append(image)
            self.idEye.append((idEye))

            if pathMask:
                imgMask_name = 'OperatorA_C{0}_S{1}_I{2}.tiff'.format(idEye,idSes,idImg)
                imgMask_path = os.path.join(pathMask,imgMask_name)
                imageMask_pil = Image.open(imgMask_path).convert('L')
                imageMask = np.array(imageMask_pil, 'uint8') # normalization
                self.MaskImages.append(imageMask)
