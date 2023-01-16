from keras.models import load_model
import cv2
import numpy as np

#load the model 

model = load_model('model-xxx.model') #change "model-xxx.model" with the model you want to use

#this model, created by Rainers, makes use of the Adaptive Boosting Algorithm (AdaBoost) in order to yield better results and accuracy.
#in other words, it is an algorithm that can detect objects in images, irrespective of their scale in image and location. 
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#set the camera

#phone cam: u must download droid cam for pc and phone

source=cv2.VideoCapture(1) # put 0 if you want to use your pc's cam

#both probabilities must be labeled (0 for "without the mask" and 1 for "with the mask").
labels_dict={0:'without_mask',1:'with_mask'}
#define the color of the limiting rectangle with the RGB values.
#1:green and 0:red
color_dict={0:(0,0,255),1:(0,255,0)}

while(True):
    #read frame by frame from the camera 
    ret,img=source.read()
    #convert them into grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detect faces
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    
    
    
    #now loop for each face
    for x,y,w,h in faces:
        #detect the ROI(region of interests), resize it and remodel it (reshape it)
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(120,120))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,120,120,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(
          img, labels_dict[label], 
          (x, y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()