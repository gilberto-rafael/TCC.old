import cv2 as cv
import numpy as np
import sys
import math

# demarca as linhas vermelhas em cima das palavras


def main():
    src = cv.imread("1.jpg")
    dst = cv.Canny(src, 50, 200)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    if True:  # HoughLinesP
        lines = cv.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 30, 10)
        a, b, c = lines.shape
        for i in range(a):
            cv.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)

    else:  # HoughLines
        lines = cv.HoughLines(dst, 1, math.pi/180.0, 50, np.array([]), 0, 0)
        if lines is not None:
            a, b, c = lines.shape
            for i in range(a):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0, y0 = a*rho, b*rho
                pt1 = (int(x0+1000*(-b)), int(y0+1000*(a)))
                pt2 = (int(x0-1000*(-b)), int(y0-1000*(a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Linhas", cdst)
    cv.waitKey(0)
    print('Done')
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
