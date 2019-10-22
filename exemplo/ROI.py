# coding: utf-8
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pyttsx3
import copy

# array da lista de resultados
results = []

voiceEngine = pyttsx3.init()
rate = voiceEngine.getProperty('rate')
volume = voiceEngine.getProperty('volume')
voice = voiceEngine.getProperty('voice')

voiceEngine.setProperty('rate', 210)
voiceEngine.setProperty('volume', 2)

# import image
image = cv2.imread("111.jpg")

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)

# binarização
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('threshold', thresh)

# dilação
kernel = np.ones((5, 20), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
# cv2.imshow('dilated', img_dilation)

# encontra contornos
# cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
cv2MajorVersion = cv2.__version__.split(".")[0]
# checagem de contornos nas linhas de acordo com a versão do CV
if int(cv2MajorVersion) >= 4:
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ordena contornos
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

# tentar ordenar de cima para baixo
# sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr[0][1]))

for i, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv2.boundingRect(ctr)
    roi = image[y:y + h, x:x + w]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if w > 15 and h > 15:
        im = Image.fromarray(roi)
        text = pytesseract.image_to_string(im)
        '''
        Estudando extras
        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))q
        '''
        print text
        voiceEngine.say(text)
        voiceEngine.runAndWait()

cv2.imshow('janela', image)
# cv2.imwrite("result.png", image)


cv2.waitKey(0)
