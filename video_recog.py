import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import math
from statistics import mean
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
prevframe=[]
timeout=MAX_Timeout=100
out = cv2.VideoWriter('output.mp4', -1, 15.0, (640,480))
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if timeout<1:
        nfaces=math.ceil(mean(prevframe))
        prevframe=[]
        timeout=MAX_Timeout
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
        timeout-=1
        prevframe.append(anterior)
        #cv2.putText(frame,'Faces Found: '+str(nfaces)+" at "+str(dt.datetime.now(),(10,450),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
        out.write(frame)
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
