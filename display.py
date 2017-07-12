import cv2
import numpy as np
from time import sleep
import math
distance = 0.0

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec=cv2.createLBPHFaceRecognizer()
rec.load("training.yml")
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
count=0

while True:
    ret,img=cam.read()
    frame = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
 
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=faceDetect.detectMultiScale(gray, 1.3, 5)
    count=count+1
    for(x,y,w,h) in faces:
        distancei = (2*3.14 * 180)/(w+h*360)*1000 + 3
       
        distance = distancei *2.54
        distance = math.floor(distance)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 3)
        id,conf=rec.predict(gray[y:y+h, x:x+w])
        if(distance<140):

          if(conf<70):
              if(id==1):
                  id="dina"
            
           
          else:
              id="unknown"  
                          
              cv2.imwrite("out/"+str(count)+".jpg",gray[y:y+h,x:x+w])

          cv2.putText(frame,str(id),(x,y+h), font, 1, (255,255,255), 3)   
          cv2.putText(frame,'Distance = ' + str(distance), (10,30),font,1,(0,0,255),2)
    cv2.imshow("Face",frame)
    if (cv2.waitKey(1) & 0xFF==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()


