#!/usr/bin/python


import cv2

gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1280, height=(int)720, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(1920, 1080)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
while(True):
    _, img = cap.read()
    dim = (720, 480)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("camera", resized)
    key = cv2.waitKey(10)
cap.release()
cv2.destroyAllWindows()