import cv2 as cv
import numpy as np
import math

def rescaleFrame(frame, scale = 0.5):
    #works for images,videos and live video
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

orig = cv.imread('Rishabh_Pant.jpg')
img = cv.imread('Rishabh_Pant.jpg')
#img = rescaleFrame(img)

#img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

lower = np.array([5, 10, 70], dtype = "uint8") 
upper = np.array([40, 175, 255], dtype = "uint8")

mask = cv.inRange(img_hsv, lower, upper)

#(cnts, _) = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#counter = 0

#for c in cnts:
 #   counter = counter+1
  #  area = cv.contourArea(c)
   # peri = cv.arcLength(c, True)
    #approx = cv.approxPolyDP(c, 0.05 * peri, True)

    #if(counter == 1):
     #   pa = area
    #elif(area>pa):
     #   pa = area
      #  ap = approx

#cv.drawContours(img, [ap], -1, (255, 0, 255), 4)

#edges = cv.Canny(mask,50,100)
#edges_org = cv.Canny(img,50,100)
#(cnts, _) = cv.findContours(edges_org.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#cv.drawContours(img, cnts, -1, (255, 0, 255), 4)
result = cv.bitwise_and(img, img, mask=mask)

cv.imshow("mask",mask)
cv.imshow('result',result)
#cv.imshow('contour',img)
cv.imwrite('mask.jpg',mask)
cv.imwrite('result.jpg',result)
#cv.imshow('edges',edges)
cv.waitKey(0)