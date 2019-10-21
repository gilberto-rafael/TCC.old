import cv2

# binarizacao da imagem

img = cv2.imread("1.jpg", 0)
cv2.imshow('original', img)

# threshold adaptativo
ret, thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('thresh1', thresh)

# contornos
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('BOX:', len(contours))

cv2.waitKey(0)
cv2.destroyAllWindows()
