import cv2
import numpy as np
import imutils

# hough varias linhas

im = cv2.imread("016.png")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
thresh_type = cv2.THRESH_BINARY_INV

bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)

# edges = cv2.Canny(gray, 80, 150, apertureSize=3)

img = im.copy()

lines = cv2.HoughLines(bin_img, 2, np.pi/180, 190)

for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow('houghlines', imutils.resize(img, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()
