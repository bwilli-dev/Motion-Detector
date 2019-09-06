import numpy as np
import cv2

def face_detect():
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img=cv2.imread("photo.jpg")
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)
    for x,y,w,h in faces:
         img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    resized=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

    cv2.imshow('Gray',resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#face_detect()

def video():
    video = cv2.VideoCapture(0)
    while(True):
        check, frame = video.read()
        print(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("capture",gray)

        key=cv2.waitKey(10)
        if key== ord('c'):
            break
    video.release()
    cv2.destroyAllWindows()
#video()
