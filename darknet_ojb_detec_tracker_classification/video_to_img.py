# Python program to explain cv2.imwrite() method 

# importing cv2 
import cv2 

# importing os module 
import os 

cap = cv2.VideoCapture("busystreet.mp4")
counter = 0
while (cap.isOpened()):
    ret, image = cap.read()
    if not ret:
        break
    filename = 100 + counter
    counter +=1
    string = "img/"+str(filename)+".jpg"
    cv2.imwrite(string, image) 
cap.release()
print('Successfully saved') 
