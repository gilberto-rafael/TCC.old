import cv2
import numpy as np

# demarca uma linha vermelha da cam com hough

def main():
    cap = cv2.imread("1.jpg")
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=5, L2gradient=True)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:

        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 800 * (-b))
            y1 = int(y0 + 800 * (a))
            x2 = int(x0 - 800 * (-b))
            y2 = int(y0 - 800 * (a))

            cv2.line(cap, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("windowName", cap)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()