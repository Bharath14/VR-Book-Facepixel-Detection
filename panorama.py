import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("bbb.jpg")
img1_res = cv2.resize(img1, (800,500))
img1_wor = cv2.cvtColor(img1_res,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("aaa.jpg")
img2_res = cv2.resize(img2, (800,500))
img2_wor = cv2.cvtColor(img2_res,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SURF_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1_wor,None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_wor,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
match = cv2.FlannBasedMatcher(index_params, search_params)
matches = match.knnMatch(descriptors1,descriptors2,k=2)

good_matches = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good_matches.append(m)
    
draw_params = dict(matchColor = (0,255,0),singlePointColor = None,flags = 2)

img3 = cv2.drawMatches(img1_wor,keypoints1,img2_wor,keypoints2,good_matches,None,**draw_params)

MIN_MATCH_COUNT = 10
if len(good_matches)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w= img1_wor.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2_wor = cv2.polylines(img2_wor,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print("Images doesn't have sufficient number of good matches" )
dst = cv2.warpPerspective(img1_res,M,(img2_res.shape[1] + img1_res.shape[1], img2_res.shape[0]))
dst[0:img2_res.shape[0], 0:img2_res.shape[1]] = img2_res

cv2.imwrite("bbb_key.jpg",cv2.drawKeypoints(img1_res,keypoints1,None))
cv2.imwrite("aaa_key.jpg",cv2.drawKeypoints(img2_res,keypoints2,None))
cv2.imwrite("lines.jpg",img3)
dst = cv2.resize(dst, (800,500))
cv2.imshow("img1",img1_wor)
cv2.imshow("img2",img2_wor)
cv2.imwrite("output.jpg",dst)
cv2.imshow("Stiched image",dst)
cv2.waitKey(0)