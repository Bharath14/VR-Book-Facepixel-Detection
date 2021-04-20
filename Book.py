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
img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
#img_blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)

#edged_blur=cv.Sobel(blur,cv.CV_64F,1,0)
#sobely=cv.Sobel(blur,cv.CV_64F,0,1)
#edged_blur=cv.bitwise_or(sobelx,sobely)

kernel = np.ones((3,3),np.uint8)
#kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
edged_blur_morph = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

sigma = 0.33
v = np.median(img)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

#edged_noblur = cv.Canny(img_blur,10,250)
edged_blur = cv.Canny(edged_blur_morph, lower,upper)

cv.imshow
#edged_blur = cv.dilate(edged_blur, kernel,iterations = 1)
#edged_blur = cv.erode(edged_blur,kernel,iterations = 1)
#edged_noblur = cv.morphologyEx(edged_blur, cv.MORPH_CLOSE, kernel)

cdst = orig
cdstP = np.copy(cdst)

lines = cv.HoughLines(edged_blur, 1, np.pi / 180, 173, None, 200, 300)

total = 0
totalp =0

if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
            total = total+1
           

linesP = cv.HoughLinesP(edged_blur, 1, np.pi / 180, 157, None, 200, 300)
    
if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            totalp = totalp+1
    
#cv.imshow("Source", img)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

#(cnts, _) = cv.findContours(edged_blur.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#total = 0

#for c in cnts:
  # approximate the contour
 # peri = cv.arcLength(c, True)
  #approx = cv.approxPolyDP(c, 0.02 * peri, True)

  # if the approximated contour has four points, then assume that the
  # contour is a book -- a book is a rectangle and thus has four vertices
  #if len(approx) == 4:
   # cv.drawContours(img, [approx], -1, (0, 255, 0), 4)
    #total += 1

print ("I found {0} books in that image".format(total))
print ("I found {0} books in that image".format(totalp))
#cv.imwrite('detect_blur.jpg',edged_blur)
#cv.imshow('img',edged_blur_morph)
#cv.imwrite('edge_morph.jpg',edged_blur_morph)
#cv.imwrite('blur.jpg',img_blur)
#cv.imwrite("Output_blur.jpg", edged_noblur)
cv.imwrite('Standard.jpg',cdst)
cv.imwrite("Probabilistic.jpg",cdstP)
cv.waitKey(0)
