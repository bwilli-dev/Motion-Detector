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
    first_frame=None
    video = cv2.VideoCapture(0)
    while(True):
        check, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(21,21),0)

        if first_frame is None:
            first_frame=gray
            continue


        delta_frame=cv2.absdiff(first_frame,gray)
        threshold=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
        threshold=cv2.dilate(threshold,None,iterations=2)

        (_,cnts,_)=cv2.findContours(threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in cnts:
            if cv2.contourArea(contour) < 10000:
                continue
            (x, y, w, h)=cv2.boundRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)


        cv2.imshow("Threshold Frame",threshold)
        cv2.imshow("capture",gray)
        print(gray)
        key=cv2.waitKey(1)
        if key== ord('c'):
            break
    video.release()
    cv2.destroyAllWindows()
video()
