# Face-And-Eyes-Detection-in-Live-Video
#Using the Open CV Module
import cv2
def face_eye_detector():

    face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_model = cv2.CascadeClassifier('haarcascade_eye.xml')

    camera = cv2.VideoCapture(0)

    while(True):
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_model.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        eyes = eye_model.detectMultiScale(gray, 1.03, 5, 0, (20,20))
        for (a, b, c, d) in eyes:
            cv2.rectangle(img, (a,b), (a+c,b+d),(0,0,255),2)

        cv2.imshow('hello', img)
        if cv2.waitKey(10) == 13:
            break

    cv2.destroyAllWindows()
    camera.release()
face_eye_detector()
