import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

names = ["Dominic", "People"]


json_file = open('testmodel.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load woeights into new model
loaded_model.load_weights("testmodel.h5")
#print("Loaded Model from disk")

#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#loss,accuracy = model.evaluate(X_test,y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)
x = imread('C:\\Users\\Dominic Mann\\Documents\\python\\image recog\\New Data SetG\\more People glasses_022.jpg',mode='L')
#x = imread('C:\\Users\\Dominic Mann\\Documents\\python\\image recog\\New Data SetG\\Me-508.png', mode='L')

x = imresize(x,(50,50))
x = x.reshape(1,50,50,1)

out = loaded_model.predict(x)
print(out)
print(np.argmax(out,axis=1))
print(names[int(np.argmax(out,axis=1))])
