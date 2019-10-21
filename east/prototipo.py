# coding: utf-8
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import pytesseract
import east


def decode_predictions(scores, geometry):
	# pega o numero de linhas e colunas para setar os boxes e possíveis correspondências de texto
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop de acordo com o número de linhas
	for y in range(0, numRows):
		# extrai as probabilidades (score), junto com as coordenadas de onde está o texto
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop de acordo com o número de colunas
		for x in range(0, numCols):
			# se nosso score nao tem probabilidade de ter palavras, ignore
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extrai o ângulo de rotaço da possibilidade e extrai seno e coseno
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# returna as boxes e seu lugares
	return (rects, confidences)


# construtor dos argumentos para efeitos de teste e rodar no terminal
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=False, help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str, help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320, help="resized image height (should be multiple of 32)")
ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

# dimensões originais do frame, novas dimensões e a proporção
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# carrega o EAST
print("[INFO] Carregando arquivo EAST...")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# se sao indicar um arquivo de video, execute com a webcam
if not args.get("video", False):
	print("[INFO] Iniciando transmissão...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture("0.mov")

# estimativa da taxa de frasnferência dos FPS
fps = FPS().start()

# loop nos frames extraídos do vídeo
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# redimensiona o frame mantendo a proporção
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	(origH, origW) = frame.shape[:2]

	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# redimensiona o frame ignorando a proporção
	frame = cv2.resize(frame, (newW, newH))

	# controi o BLOB do frame e altera o input do modelo para obter duas saídas
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# decodifica previsões, aplica non-maxima suppression para evitar sobrescrever caixas
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# array da lista de resultados
	results = []

	# loop nas boxes
	for (startX, startY, endX, endY) in boxes:
		# mantém escala das boxes de acordo com a proporção
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# desenha as boxes no vídeo
		cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 0, 0), 2)

		''' EXTRAS

		# in order to obtain a better OCR of the text we can potentially
		# apply a bit of padding surrounding the bounding box -- here we
		# are computing the deltas in both the x and y directions
		dX = int((endX - startX) * args["padding"])
		dY = int((endY - startY) * args["padding"])

		# apply padding to each side of the bounding box, respectively
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		# extract the actual padded ROI
		roi = orig[startY:endY, startX:endX]

		# in order to apply Tesseract v4 to OCR text we must supply
		# (1) a language, (2) an OEM flag of 4, indicating that the we
		# wish to use the LSTM neural net model for OCR, and finally
		# (3) an OEM value, in this case, 7 which implies that we are
		# treating the ROI as a single line of text
		config = ("-l eng --oem 2 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)

		# add the bounding box coordinates and OCR'd text to the list
		# of results
		results.append(((startX, startY, endX, endY), text))

	# sort the results bounding box coordinates from top to bottom
	results = sorted(results, key=lambda r: r[0][1])
	# FIM EXTRAS '''

	# atualiza contador de FPS
	fps.update()
	# mostra o frame de saída
	cv2.imshow("Janela", orig)
	key = cv2.waitKey(1) & 0xFF

	# loop nos resultados
	for ((startX, startY, endX, endY), text) in results:
		# exibe palavras reconhecidas
		print("========")
		text = text.encode('utf-8')
		print("{}\n".format(text))
		# strip out non-ASCII text so we can draw the text on the image
		# using OpenCV, then draw the text and a bounding box surrounding
		# the text region of the input image
		# text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		# output = orig.copy()
		# cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
		# cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

		# exibe saída com não reconhecidos
		#cv2.imshow("Tela 2", output)

	# sai do loop se pressionar 'q'
	if key == ord("q"):
		break

# mostra informações da execução
fps.stop()
print("[INFO] tempo estimado: {:.2f}".format(fps.elapsed()))
print("[INFO] aprox. FPS: {:.2f}".format(fps.fps()))

# libera ponteiro da webcam
if not args.get("video", False):
	vs.stop()

# libera ponteiro do arquivo de video
else:
	vs.release()

# fecha janelas
cv2.destroyAllWindows()

