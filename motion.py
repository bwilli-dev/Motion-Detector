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
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    while True:
    	ret,frame = video.read()
    	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    	if ret == False:
    		continue
    	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    	if len(faces) == 0:
    		continue
    	for face in faces:
    		x,y,w,h = face
    		offset = 10
    		face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
    		face_selection = cv2.resize(face_offset,(100,100))
    		cv2.imshow("Face", face_selection)
    		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    	cv2.imshow("faces",frame)
    	key_pressed = cv2.waitKey(1) & 0xFF
    	if key_pressed == ord('q'):
    		break
    video.release()
    cv2.destroyAllWindows()

video()
