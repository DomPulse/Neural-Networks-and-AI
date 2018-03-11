import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = "women"
f = 0
for file in os.listdir(path):
    img = cv2.imread(path + '\\' + file)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = img[y:y+h, x:x+w]
        f += 1
        cv2.imwrite('Faces\\' + path + str(f) + '.png', crop_img)
        cv2.circle(img, (int(x+(w/2)), int(y+(h/2))), int((w+h)/4), (0, 0, 255), 2)
    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
