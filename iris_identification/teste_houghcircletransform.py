import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('Figure_3.png',0)
img = cv2.imread('databases/CASIA-IrisV4-Lamp-100/045/R/S2045R05.jpg',0)

print(img)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=70,param2=20,minRadius=30,maxRadius=65)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
print("ok")
plt.imshow(cimg)
plt.pause(5)
plt.close()