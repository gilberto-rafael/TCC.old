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

# importa imagem
image = cv2.imread('111.jpg')

# escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# binario
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
# dilatação
kernel = np.ones((5, 30), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

# encontra contornos
tst = img_dilation.copy()

if cv2.getVersionMajor() in [2, 4]:
    # OpenCV 2, OpenCV 4
    contour, hier = cv2.findContours(tst, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
else:
    # OpenCV 3
    im2, contour, hier = cv2.findContours(tst, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# ordena contornos de cima para baixo
sorted_ctrs = sorted(contour, key=lambda ctr: cv2.boundingRect(ctr)[1])


for i, ctr in enumerate(sorted_ctrs):
    # pega as boxes
    x, y, w, h = cv2.boundingRect(ctr)

    # ROI
    roi = image[y:y+h, x:x+w]

    #cv2.imshow('segment no:'+str(i), roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.waitKey(0)

    # falar
    if w > 15 and h > 15:
        im = Image.fromarray(roi)
        text = pytesseract.image_to_string(im)
        print text
        voiceEngine.say(text)
        voiceEngine.runAndWait()

cv2.imshow('Boxes nas linhas', image)
cv2.waitKey(0)
