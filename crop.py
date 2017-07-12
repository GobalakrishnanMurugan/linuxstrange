import numpy as np
import cv2
import os
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
id=raw_input("\nenter ur  id:")
count=0
while 1:
    ret,img=cam.read()
    frame = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    faces=faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        count=count+1
        cv2.imwrite("faces/user."+str(id)+"."+str(count)+".jpg",hist[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
    cv2.imshow('img',frame)
    cv2.waitKey(1)
    if (count>20):
       break

cam.release()
cv2.destroyAllWindows()
