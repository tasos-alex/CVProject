import cv2
import numpy as np


#calling all  the filenames of the images we will use
img0 = cv2.imread('yard-00.png')
img1 = cv2.imread('yard-01.png')
img2 = cv2.imread('yard-02.png')
img3 = cv2.imread('yard-03.png')

image0 = img0.copy()
image1 = img1.copy()
image2 = img2.copy()
image3 = img3.copy()


#cross checking function
def cross_checking(matches1, matches2):
    cross_check_matches = []
    for match1 in matches1:
        for match2 in matches2:
            if match1.queryIdx == match2.trainIdx and match1.trainIdx == match2.queryIdx:
                cross_check_matches.append(match1)
    return cross_check_matches


#-----------------------------------------------------
#-----------------------------------------------------
#------------------SURF-------------------------------
#-----------------------------------------------------
#-----------------------------------------------------


#calling surf object
surf = cv2.xfeatures2d.SURF_create()


kp0, des0 = surf.detectAndCompute(img0.copy(), None)
imgkp0 = cv2.drawKeypoints(img0.copy(), kp0, img0.copy())

cv2.imshow('surf keypoints 0' , imgkp0)
cv2.waitKey(0)

kp1, des1 = surf.detectAndCompute(img1.copy(), None)
imgkp1 = cv2.drawKeypoints(img1.copy(), kp1, img1.copy())

cv2.imshow('surf keypoints 1' , imgkp1)
cv2.waitKey(0)

kp2, des2 = surf.detectAndCompute(img2.copy(), None)
imgkp2 = cv2.drawKeypoints(img2.copy(), kp2, img2.copy())

cv2.imshow('surf keypoints 2' , imgkp2)
cv2.waitKey(0)

kp3, des3 = surf.detectAndCompute(img3.copy(), None)
imgkp3 = cv2.drawKeypoints(img3.copy(), kp3, img3.copy())

cv2.imshow('surf keypoints 3' , imgkp3)
cv2.waitKey(0)


#BF matcher for matching WITHOUT CROSSCHECKING
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

#matching the keypoints found with surf for the two outer right images
#and using the cross check to keep the common matches
surf_matches01 = bf.match(des0, des1)
surf_matches10 = bf.match(des1, des0)

cross_check01 = cross_checking(surf_matches10, surf_matches01)

#matching the keypoints found with sift for the two outer left images
#and using the cross check to keep the common matches
surf_matches23 = bf.match(des2, des3)
surf_matches32 = bf.match(des3, des2)

cross_check23 = cross_checking(surf_matches32, surf_matches23)


#creating and showing images with the matching keypoints
dimg1 = cv2.drawMatches(img1.copy(), kp1, img0.copy(), kp0, cross_check01, None)
dimg2 = cv2.drawMatches(img3.copy(), kp3, img2.copy(), kp2, cross_check23, None)
# dimg1 = cv2.drawMatches(img1.copy(), kp1, img0.copy(), kp0, cross_check01, None)

cv2.namedWindow('matches01', cv2.WINDOW_NORMAL)
cv2.namedWindow('matches23', cv2.WINDOW_NORMAL)
# cv2.namedWindow('matches12', cv2.WINDOW_NORMAL)


cv2.imshow('matches01', dimg1)
cv2.imshow('matches23', dimg2)
# cv2.imshow('matches12', dimg2)


cv2.waitKey(0)

img_pt0 = []
img_pt1 = []
img_pt2 = []
img_pt3 = []

#queryIdx από την αριστερή εικόνα και trainIdx από την δεξιά για το δεξιό ζεύγος εικόνων
for x in cross_check01:
    img_pt1.append(kp1[x.queryIdx].pt)
    img_pt0.append(kp0[x.trainIdx].pt)

img_pt0 = np.array(img_pt0)
img_pt1 = np.array(img_pt1)


#queryIdx από την αριστερή εικόνα και trainIdx από την δεξιά για το αριστερό ζεύγος εικόνων
for x in cross_check23:
    img_pt3.append(kp3[x.queryIdx].pt)
    img_pt2.append(kp2[x.trainIdx].pt)

img_pt2 = np.array(img_pt2)
img_pt3 = np.array(img_pt3)

#τα ορίσματα πρέπει να μπουν πρώυα η δεξιά εικόνα μετά η αριστερά για τη δημιουργία μάσκας
M01, mask01 = cv2.findHomography(img_pt0, img_pt1, cv2.RANSAC)

M23, mask23 = cv2.findHomography(img_pt2, img_pt3, cv2.RANSAC)



#τα ορίσματα πρέπει να μπουν πρώτα η δεξιά εικόνα μετά η αριστερά
merged1 = cv2.warpPerspective(img0, M01, (img1.shape[1] + 500, img1.shape[1] + 50))

merged2 = cv2.warpPerspective(img2, M23, (img3.shape[1] + 500, img3.shape[1] + 50))



#ορίσματα μόνο από αριστερή εικόνα
merged1[0: img1.shape[0], 0: img1.shape[1]] = img1

merged2[0: img3.shape[0], 0: img3.shape[1]] = img3




cv2.namedWindow('panorama01', cv2.WINDOW_NORMAL)
cv2.imshow('panorama01', merged1)

cv2.namedWindow('panorama23', cv2.WINDOW_NORMAL)
cv2.imshow('panorama23', merged2)

cv2.waitKey(0)

#τελικά συγχωνεύουμε τα δυο παραγόμενα πανοράματα για να φτιάξουμε το τελικό

#υπολογισμός keypoint-descriptor για το δεξί ζεύγος
kp01, des01 = surf.detectAndCompute(merged1.copy(), None)
imgkp01 = cv2.drawKeypoints(merged1.copy(), kp01, merged1.copy())

#υπολογισμός keypoint-descriptor για το αριστερό ζεύγος
kp23, des23 = surf.detectAndCompute(merged2.copy(), None)
imgkp23 = cv2.drawKeypoints(merged2.copy(), kp23, merged2.copy())

#mathcing keypoints & cross checking
surf_matches0123 = bf.match(des01, des23)
surf_matches2301 = bf.match(des23, des01)

cross_check0123 = cross_checking(surf_matches2301, surf_matches0123)

dimg3 = cv2.drawMatches(merged2.copy(), kp23, merged1.copy(), kp01, cross_check0123, None)

cv2.namedWindow('matches0123', cv2.WINDOW_NORMAL)
cv2.imshow('matches0123', dimg3)
cv2.waitKey(0)

img_pt01 = []
img_pt23 = []

for x in cross_check0123:
    img_pt23.append(kp23[x.queryIdx].pt)
    img_pt01.append(kp01[x.trainIdx].pt)

img_pt01 = np.array(img_pt01)
img_pt23 = np.array(img_pt23)

M0123, mask0123 = cv2.findHomography(img_pt01, img_pt23, cv2.RANSAC)

merged3 = cv2.warpPerspective(merged1, M0123, (merged2.shape[1] + 500, merged2.shape[1] + 50))

merged3[0: merged2.shape[0], 0: merged2.shape[1]] = merged2

cv2.namedWindow('panorama0123', cv2.WINDOW_NORMAL)
cv2.imshow('panorama0123', merged3)
cv2.waitKey(0)