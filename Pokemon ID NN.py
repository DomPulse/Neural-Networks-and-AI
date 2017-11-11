import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

names = ["Gen1", "Gen2", "Gen3", "Gen4", "Gen5", "Gen6", "Gen7"]

json_file = open('PokeModel.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load woeights into new model
loaded_model.load_weights("PokeModel.h5")
#print("Loaded Model from disk")

#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#loss,accuracy = model.evaluate(X_test,y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)

#x = imread('Personal Test\Dex_143.png',mode='L')
#x = imread('Personal Test\Dex_212.png',mode='L')
#x = imread('Personal Test\Dex_306.png',mode='L')
#x = imread('Personal Test\Dex_456.png',mode='L')
#x = imread('Personal Test\Dex_578.png',mode='L')
#x = imread('Personal Test\Dex_691.png',mode='L')
x = imread('Personal Test\Dex_775.png',mode='L')

x = imresize(x,(40,40))
x = x.reshape(1,40,40,1)

out = loaded_model.predict(x)
print(out)
print(np.argmax(out,axis=1))
print(names[int(np.argmax(out,axis=1))])
