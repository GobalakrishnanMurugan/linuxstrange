import os
import cv2
import numpy as np
from PIL import Image
recognizer=cv2.createLBPHFaceRecognizer()
path='faces/'

def getmagewithid(path):
   imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
   faces=[]
   ids=[]
   for imagepath in imagepaths:
      faceimg=Image.open(imagepath).convert('L')
      facenp=np.array(faceimg, 'uint8')
      id=int(os.path.split(imagepath)[-1].split('.')[1])
      faces.append(facenp)
      print id
      ids.append(id)
      cv2.imshow('training',facenp)
      cv2.waitKey(10)
   return ids, faces

ids,faces=getmagewithid(path) 
recognizer.train(faces,np.array(ids))
recognizer.save('training.yml')
cv2.destroyAllWindows()   
