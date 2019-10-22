import cv2
import numpy as np
#import image
image = cv2.imread('111.jpg')

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#binary
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#dilation
kernel = np.ones((5, 100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

#find contours
tst = img_dilation.copy()
# im2, ctrs, hier = cv2.findContours(tst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if cv2.getVersionMajor() in [2, 4]:
    # OpenCV 2, OpenCV 4 case
    contour, hier = cv2.findContours(tst, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
else:
    # OpenCV 3 case
    image, contour, hier = cv2.findContours(tst, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#sort contours
sorted_ctrs = sorted(contour, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    # cv2.imshow('segment no:'+str(i), roi)
    cv2.rectangle(image,(x,y),( x + w, y + h ),(0,0,255),2)
    cv2.waitKey(0)

cv2.imshow('Boxes nas palavras', image)
cv2.waitKey(0)
