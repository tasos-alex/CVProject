import cv2
import numpy as np

#calling all  the filenames of the images we will use
img0 = cv2.imread('yard-00.png')
img1 = cv2.imread('yard-01.png')
img2 = cv2.imread('yard-02.png')
img3 = cv2.imread('yard-03.png')

#creating a list with our images
images = [img0, img1, img2, img3]

# cv2.imshow('start image', img)
# cv2.waitKey(0)

#grayscaling our images
#(this step is optional because the images are already grayscaled)
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

#creating two lists with the grayscaled images
grays = [gray0, gray1, gray2, gray3]
grays2 = grays

# cv2.imshow('gray image', gray)
# cv2.waitKey(0)

#-----------------------------------------------------
#-----------------------------------------------------
#------------------SIFT-------------------------------
#-----------------------------------------------------
#-----------------------------------------------------


#sift = cv2.SIFT()
#calling sift object
sift = cv2.xfeatures2d.SIFT_create()

#creating a list to save the keypoints of each image
#creating a list to save the images with keypoints shown
sift_keypoints = []
sift_descriptors = []
sift_kp_images = []

#detection of keypoints in every gray image
#drawing the keypoints on the correspoding image
#saving the new image in the list
for gray in grays:
    kp, des = sift.detectAndCompute(gray, None)
    sift_keypoints.append(kp)
    sift_descriptors.append(des)
    img = cv2.drawKeypoints(gray, kp, gray)
    sift_kp_images.append(img)

# cv2.imshow('keypoint image', img)
# cv2.waitKey(0)

#creating jpg images with the keypoints drawn, saving and showing them
for index, kp_image in enumerate(sift_kp_images):
    ind = str(index)
    txt = 'sift_keypoints' + ind + '.jpg'
    cv2.imwrite(txt, kp_image)
    cv2.imshow('sift keypoints' +ind , kp_image)
    cv2.waitKey(0)






#-----------------------------------------------------
#-----------------------------------------------------
#------------------SURF-------------------------------
#-----------------------------------------------------
#-----------------------------------------------------

#surf object
surf = cv2.xfeatures2d.SURF_create(500)
#surf.hessianThreshold = 50000

#creating a list to save the keypoints of each image
#creating a list to save the images with keypoints shown
surf_keypoints = []
surf_descriptors = []
surf_kp_images = []


#detection of keypoints in every gray image
#drawing the keypoints on the correspoding image
#saving the new image in the list
for gray in grays2:
    kp, des = surf.detectAndCompute(gray, None)
    surf_keypoints.append(kp)
    surf_descriptors.append(des)
    img = cv2.drawKeypoints(gray, kp, None, (255, 0, 0), 4)
    surf_kp_images.append(img)
    #plt.imshow(img), plt.show()

#creating jpg images with the keypoints drawn, saving and showing them
for index, kp_image in enumerate(surf_kp_images):
    ind = str(index)
    txt = 'surf_keypoints' + ind + '.jpg'
    cv2.imwrite(txt, kp_image)
    cv2.imshow('surf keypoints' +ind , kp_image)
    cv2.waitKey(0)
