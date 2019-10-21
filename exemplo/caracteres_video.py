from PIL import Image
import cv2
import pytesseract
import pyttsx3

# leitura dos caracteres no video
# contador de frames para nao ler todos os frames
indiceFrame = 0
valorAnterior = "a"

video = cv2.VideoCapture(0)

engine = pyttsx3.init()

while True:
    indiceFrame += 1
    ret, frameRGB = video.read()
    imagem = Image.fromarray(frameRGB)
    if indiceFrame == 20:
        indiceFrame = 0
        valorAtual = pytesseract.image_to_string(imagem)

        if valorAtual != valorAnterior:

            aux = valorAtual
            remov = aux.lstrip(valorAnterior)
            valorAnterior = valorAtual
            print remov
            engine.say(remov)
            engine.runAndWait()

    cv2.imshow("Video",	frameRGB)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
