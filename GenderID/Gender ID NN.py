import cv2
import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow

#load and initialize model

json_file = open('GenderID\GenderID.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("GenderID\GenderID.h5")
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#initialize open cv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
colors = [(255, 230, 44), (135, 44, 255)]

def SID(x):
    x = imresize(x,(150,150))
    x = x.reshape(1,150,150,3)

    out = loaded_model.predict(x)
    return(np.argmax(out,axis=1)[0])

a = 0
S = 0
while True:
    ret, img = cap.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    f = 1
    for (x, y, w, h) in faces:
        crop_img = img[y:y+h, x:x+w]
        if x % 25 == 0:
            S = SID(crop_img)
        color = colors[S]
        cv2.circle(img, (int(x+(w/2)), int(y+(h/2))), int((w+h)/4), color, 2)
        coor = "X: " + str(x+(w/2)) + " Y: " + str(y+(h/2))
        cv2.putText(img, coor, (x, y-20), font, 0.8, color, 2, cv2.LINE_AA)
        print("Face", f, ": X: ", x+(w/2), ", Y:" , y+(h/2))
        f+=1
    a+=1
    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
