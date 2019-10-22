import cv2
import numpy as np
import pytesseract
from PIL import Image
import pyttsx3

engine = pyttsx3.init()

def get_string(img_path):
    # Ler imagem com opencv
    img = cv2.imread(img_path)

    # Converter para cinza
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # aplicar dilatacao e erosao para remover ruido, melhora reconhecimento antes do opencv

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # salva imagem sem ruido
    cv2.imwrite("ruido_removido.png", img)

    #  aplica threshold para obter imagem PB
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # salva imagem apos threshold
    cv2.imshow("threshold", img)

    # reconhece texto com pytesseract
    result = pytesseract.image_to_string(img)

    # remove o temp
    #os.remove(temp)

    engine.say(result)
    engine.runAndWait()
    return result


print get_string("111.jpg")

