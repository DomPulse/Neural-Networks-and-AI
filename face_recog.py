import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    f = 1
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (119, 119, 255), 2)
        print("Face", f, ": X: ", (x+w)/2, ", Y:" , (y+h)/2)
        f+=1
    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

#most credit goes to sentdex in his video: https://www.youtube.com/watch?v=88HdqNDQsEk
