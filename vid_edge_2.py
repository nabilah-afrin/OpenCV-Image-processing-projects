import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)
  
while(True):

    var , frame = cam.read()
    frame = cv2.resize(frame, (480, 360), interpolation = cv2.INTER_AREA)

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    Roberts_3d = cv2.cvtColor(Roberts, cv2.COLOR_GRAY2BGR);

    canny = cv2.Canny(frame,100,100)
    canny_3d = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR);

    lapi = np.uint8(cv2.Laplacian(frame, cv2.CV_64F));

    hw1 = np.hstack((frame, canny_3d))
    hw2 = np.hstack((lapi, Roberts_3d))

    vw = np.vstack((hw1, hw2))

    cv2.imshow("Original |  Canny Edges  |  Laplacian Edges | Roberts", vw)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
