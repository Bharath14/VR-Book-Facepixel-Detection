import cv2 as cv
import numpy as np
import math
def rescaleFrame(frame, scale = 0.2):
    #works for images,videos and live video
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

orig = cv.imread('Bookshelf.jpg')
orig = rescaleFrame(orig)
img = cv.imread('Bookshelf.jpg')
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
img = rescaleFrame(img)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#img = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)

#edged_blur=cv.Sobel(blur,cv.CV_64F,1,0)
#sobely=cv.Sobel(blur,cv.CV_64F,0,1)
#edged_blur=cv.bitwise_or(sobelx,sobely)
sigma = 0.33
v = np.median(img)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

kernel = np.ones((10,10),np.uint8)
#edged_blur = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

#edged_noblur = cv.Canny(gray,10,250)
edged_blur = cv.Canny(img, lower,upper)

(cnts, _) = cv.findContours(edged_blur.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
total = 0

cv.drawContours(orig, cnts, -1, (0, 255, 0), 4)
for c in cnts:
  # approximate the contour
  area = cv.contourArea(c)
  peri = cv.arcLength(c, True)
  approx = cv.approxPolyDP(c, 0.03 * peri, True)

   #if the approximated contour has four points, then assume that the
   #contour is a book -- a book is a rectangle and thus has four vertices
  #if area>100:
  #cv.drawContours(orig, [approx], -1, (0, 255, 0), 4)
  total = total+1

#total = len(cnts)

print ("I found {0} books in that image".format(total))
#cv.imshow("blur",edged_blur)
cv.imwrite('contour.jpg',orig)
cv.imshow("output",orig)
cv.waitKey(0)