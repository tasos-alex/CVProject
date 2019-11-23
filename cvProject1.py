import cv2           #OpenVB version 4.1.1
import numpy as np   #Numpy για αριθμητικές πράξεις με πίνακες

image_filename = 'NF6.png'
noisy_image_filename = 'N6.png'

#routine for denoising
def median_filter(img):
    img2 = np.copy(img)#np.zeros(shape = img.shape)
    for i in range(img.shape[0]-2):
        for j in range(img.shape[1]-2):
            temp = [img[i][j], img[i][j+1], img[i][j+2], img[i+1][j], img[i+1][j+1], img[i+1][j+2], img[i+2][j], img[i+2][j+1], img[i+2][j+2]]
            temp.sort()
            img2[i+1][j+1] = temp[4]
    return img2


#preparing images
noisy_image = cv2.imread(noisy_image_filename, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
# print(noisy_image.shape[1])
# print(noisy_image[3,345])
#denoised_image = np.zeros(shape = noisy_image.shape)



denoised = median_filter(noisy_image)
print(image)
cv2.namedWindow('main') #Δημιουργία παραθύρου με όνομα "main"
cv2.imshow('main', image) #Προβολή εικόνας στο παράθυρο main
cv2.namedWindow('main2') #Δημιουργία παραθύρου με όνομα "main"
cv2.imshow('main2', noisy_image) #Προβολή εικόνας στο παράθυρο main
cv2.namedWindow('main3') #Δημιουργία παραθύρου με όνομα "main"
cv2.imshow('main3', denoised) #Προβολή εικόνας στο παράθυρο main
cv2.waitKey(0)

# denoised2 = cv2.Sobel(denoised, cv2.CV_8UC1, 0, 1)
# cv2.namedWindow('main4') #Δημιουργία παραθύρου με όνομα "main"
# cv2.imshow('main4', denoised2) #Προβολή εικόνας στο παράθυρο main
# cv2.waitKey(0)
denoised3 = median_filter(denoised)
cv2.namedWindow('main5') #Δημιουργία παραθύρου με όνομα "main"
cv2.imshow('main5', denoised3) #Προβολή εικόνας στο παράθυρο main
cv2.waitKey(0)

ret, thresh_img = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
cv2.namedWindow('main6') #Δημιουργία παραθύρου με όνομα "main"
cv2.imshow('main6', thresh_img) #Προβολή εικόνας στο παράθυρο main
cv2.waitKey(0)


#1st task
edged=cv2.Canny(thresh_img, 30, 200)
cv2.imshow('canny edges', edged)
cv2.waitKey(0)

edged.copy()
contours, hierarchy=cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.imshow('canny edges after contouring', edged)
cv2.waitKey(0)

print(contours)
print('Numbers of contours found=' + str(len(contours)))

cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow('contours', image)
cv2.waitKey(0)
#cv2.destroyAllWindows()


#2nd task
pixels_of_cells = []
for i in range(len(contours)):
    pixels_of_cells .append(contours[i].shape[0]*contours[i].shape[1])
print(pixels_of_cells)
print(len(pixels_of_cells))

#3rd task
sum_box = np.zeros(shape=image.shape)
length = image.shape[1]
height = image.shape[0]
for i in range (height):
    for j in range(length):
        if (i-1 >= 0) & (j-1 >= 0):
            sum_box[i][j] = image[i][j] + sum_box[i-1][j] + sum_box[i][j-1] - sum_box[i-1][j-1]
        elif (i-1 >= 0) & (j-1 < 0):
            sum_box[i][j] = image[i][j] + sum_box[i-1][j]
        elif (i-1 < 0) & (j-1 <= 0):
            sum_box[i][j] = image[i][j] + sum_box[i][j-1]
        else:
            sum_box[i][j] = image[i][j]
print(sum_box)
# print(sum(image[0:2][0:5]))
cv2.imshow('sumbox', sum_box)
cv2.imshow('image again', image)
cv2.waitKey(0)
