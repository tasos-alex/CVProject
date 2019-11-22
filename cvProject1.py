import cv2           #απαραίτητες βιβλιοθήκες
import numpy as np
import math

NFimg_filename = 'NF6.png'
Nimg_filename = 'N6.png'
NFimg = cv2.imread(NFimg_filename)
Nimg = cv2.imread(Nimg_filename)

#συνάρτηση αφαίρεσης θορύβου
def median_filter(img):
    img2 = np.copy(img)
    for i in range(img.shape[0]-2):
        for j in range(img.shape[1]-2):
            temp = [img[i][j], img[i][j+1], img[i][j+2], img[i+1][j], img[i+1][j+1], img[i+1][j+2], img[i+2][j], img[i+2][j+1], img[i+2][j+2]]
            temp.sort()
            img2[i+1][j+1] = temp[4]
    return img2

#εφαρμόζω grayscale
NimgGray = cv2.cvtColor(Nimg, cv2.COLOR_BGR2GRAY)
NFimgGray = cv2.cvtColor(NFimg, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Noise Free')
cv2.imshow('Noise Free', NFimgGray)
cv2.namedWindow('Salt and Pepper')
cv2.imshow('Salt and Pepper', NimgGray)
cv2.waitKey(0)


#κρατάω αντίγραφα γιατί οι συναρτήσεις της opencv μεταποιούν την εικόνα που δέχονται σαν όρισμα
NimgGray1 = NimgGray.copy()
NFimgGray1 = NFimgGray.copy()

#οι διαστάσεις και των δύο εικόνων είναι ίδιες
length = NimgGray.shape[1] #αποθηκέυω τον αριθμό των στηλών για μετέπειτα χρήση
height1 = NimgGray.shape[0] #αποθηκέυω τον αριθμό των γραμμών για μετέπειτα χρήση

#εφαρμόζω το φίλτρο που δημιούργησα
denoise1 = median_filter(NimgGray1)

cv2.namedWindow('Denoised 1 time')
cv2.imshow('Denoised 1 time', denoise1)
cv2.waitKey(0)

denoise2 = median_filter(denoise1) #χρειάζεται κι άλλη βελτίωση

cv2.namedWindow('Denoised 2 times')
cv2.imshow('Denoised 2 times', denoise2)
cv2.waitKey(0)

denoise3 = median_filter(denoise2) #χρειάζεται κι άλλη βελτίωση

cv2.namedWindow('Denoised 3 times')
cv2.imshow('Denoised 3 times', denoise3)
cv2.waitKey(0)


#δημιουργία πίνακα με άθροισμα όλων των αριστερά πάνω πίξελ
sum_box1 = np.zeros_like(denoise3)

sum_box2 = np.zeros_like(NFimgGray1)

#δημιουργία πίνακα με τιμή κάθε πίξελ ίση με άθροισμα τιμών των πάνω αριστερά πίξελ
for i in range(height1):
    for j in range(length):
        if (i-1 >= 0) & (j-1 >= 0):
            sum_box1[i][j] = denoise3[i][j] + sum_box1[i - 1][j] + sum_box1[i][j - 1] - sum_box1[i - 1][j - 1]
        elif (i-1 >= 0) & (j-1 < 0):
            sum_box1[i][j] = denoise3[i][j] + sum_box1[i - 1][j]
        elif (i-1 < 0) & (j-1 <= 0):
            sum_box1[i][j] = denoise3[i][j] + sum_box1[i][j - 1]
        else:
            sum_box1[i][j] = denoise3[i][j]

for i in range(height1):
    for j in range(length):
        if (i-1 >= 0) & (j-1 >= 0):
            sum_box2[i][j] = NFimgGray1[i][j] + sum_box2[i - 1][j] + sum_box2[i][j - 1] - sum_box2[i - 1][j - 1]
        elif (i-1 >= 0) & (j-1 < 0):
            sum_box2[i][j] = NFimgGray1[i][j] + sum_box2[i - 1][j]
        elif (i-1 < 0) & (j-1 <= 0):
            sum_box2[i][j] = NFimgGray1[i][j] + sum_box2[i][j - 1]
        else:
            sum_box2[i][j] = NFimgGray1[i][j]

cv2.namedWindow('Sum Box1')
cv2.imshow('Sum Box1', sum_box1)
cv2.waitKey(0)

cv2.namedWindow('Sum Box2')
cv2.imshow('Sum Box2', sum_box2)
cv2.waitKey(0)

#δημιουργία δυαδικής εικόνας με κατώφλι 50 γιατί αλλιώς τα εσωτερικά σημεία των κυττάρων δημιουργούν πρόβλημα
ret, thresh_img1 = cv2.threshold(denoise3.copy(), 50, 255, cv2.THRESH_BINARY)
ret, thresh_img2 = cv2.threshold(NFimgGray1.copy(), 50, 255, cv2.THRESH_BINARY)


cv2.namedWindow('Binary of denoised3')
cv2.imshow('Binary of denoised3', thresh_img1)
cv2.waitKey(0)

cv2.namedWindow('Binary of NF')
cv2.imshow('Binary of NF', thresh_img2)
cv2.waitKey(0)


#εφαρμογή open για σαφή διάκριση αντικειμένων στην εικόνα
strel = np.ones((21,21), np.uint8)
opened_thresh1 = cv2.morphologyEx(thresh_img1, cv2.MORPH_OPEN, strel)
opened_thresh2 = cv2.morphologyEx(thresh_img2, cv2.MORPH_OPEN, strel)


cv2.namedWindow('Opened1', )
cv2.imshow('Opened1', opened_thresh1)
cv2.waitKey(0)

cv2.namedWindow('Opened2', )
cv2.imshow('Opened2', opened_thresh2)
cv2.waitKey(0)

#η findContours θα εντωπύσει κάθε κύτταρο
contours1, _ = cv2.findContours(opened_thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, _ = cv2.findContours(opened_thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


#δημιουργία πίνακα για αποθήκευση εικονοστοιχείων κάθε αντικειμένου
pixelsOfCells1 = []
pixelsOfCells2 = []


meanGrey1 = []
meanGrey2 = []


#αποθήκευση της θέσης των αντικειμένων που ανήκουν στο περίγραμμα
indx = 0
numOfContours = len(contours1)
while indx < numOfContours:
    for pixels in contours1[indx]:
        if pixels[0][0] == 1 or pixels[0][0] == denoise3.shape[1]-2 or pixels[0][1] == 1 or pixels[0][1] == denoise3.shape[0]-1:
            del contours1[indx]
            indx = indx-1
            numOfContours = len(contours1)
            break
    indx = indx+1

indx = 0
numOfContours = len(contours2)
while indx < numOfContours:
    for pixels in contours2[indx]:
        if pixels[0][0] == 1 or pixels[0][0] == NFimgGray1.shape[1]-2 or pixels[0][1] == 1 or pixels[0][1] == NFimgGray1.shape[0]-1:
            del contours2[indx]
            indx = indx-1
            numOfContours = len(contours2)
            break
    indx = indx+1

#σχεδίαση ορθογωνίων τμημάτων που περικλύουν τα ορθά αντικείμενα
count = 0
for valid_contour in contours1:
    x, y, w, h = cv2.boundingRect(valid_contour)
    cv2.rectangle(denoise3, (x, y), (x + w, y + h), (255, 255, 255), 2)
    text = str(count)
    cv2.putText(denoise3, text, (x+math.floor(w/2), y+math.floor(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 5)
    count += 1
    pixelsOfCells1.append(w * h)
    meanGrey1.append(sum_box1[y + h][x + w] + sum_box1[y][x] - sum_box1[y + h - 1][x + w] - sum_box1[y + h][x + w - 1])

count = 0
for valid_contour in contours2:
    x, y, w, h = cv2.boundingRect(valid_contour)
    cv2.rectangle(NFimgGray1, (x, y), (x + w, y + h), (255, 255, 255), 2)
    text = str(count)
    cv2.putText(NFimgGray1, text, (x+math.floor(w/2), y+math.floor(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 5)
    count += 1
    pixelsOfCells2.append(w * h)
    meanGrey2.append(sum_box2[y + h][x + w] + sum_box2[y][x] - sum_box2[y + h - 1][x + w] - sum_box2[y + h][x + w - 1])

print("Number of cells in N:", len(pixelsOfCells1))
print("List with the pixels of every cell in N:", pixelsOfCells1)

print("Number of cells in NF:", len(pixelsOfCells2))
print("List with the pixels of every cell in NF:", pixelsOfCells2)

cv2.imshow('Bounding rect in N', denoise3)
cv2.waitKey(0)

cv2.imshow('Bounding rectin NF', NFimgGray1)
cv2.waitKey(0)

print("MeanGray list for every cell in N:", meanGrey1)
print("MeanGray list for every cell in NF:", meanGrey2)
