from __future__ import print_function
#simplified interface for building models
import keras
#our handwritten character labeled dataset
#from keras.datasets import mnist
#because our models are simple
from keras.models import Sequential
#dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
#into respective layers
from keras.layers import Dense, Dropout, Flatten
#for convolution (images) and pooling is a technique to help choose the most relevant features in an image
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#this part is to make the data sets, credit goes to https://github.com/anujshah1003

from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 40, 40

# number of channels
img_channels = 1

path1 = "List of Pokemon by index number"    #path of folder of images
path2 = "List of Pokemon by index numberG"  #path of folder to save images

listing = os.listdir(path1)
num_samples=len(listing)
print (num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
                #need to do some more processing here
    gray.save(path2 +'\\' +  file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open("List of Pokemon by index numberG" + '\\'+ imlist[0])) # open one image to get size
x,y = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open("List of Pokemon by index numberG"+ '\\' + im2)).flatten()
              for im2 in imlist],'f')


label=np.ones((num_samples,),dtype = int)
label[0:149]=0
label[150:247]=1
label[248:382]=2
label[383:496]=3
label[497:653]=4
label[654:724]=5
label[725:805]=6

data,label = shuffle(immatrix, label, random_state=4)
train_data = [data,label]

img=immatrix[167].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)

# number of output classes
nb_classes = 7

(x, y) = (train_data[0],train_data[1])

# STEP 1: split X and y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#this concludes the data set protion of the code

#mini batch gradient descent ftw
batch_size = 128
#4 different classes of image: Me, Danny Devito, and Other
classes = nb_classes
epochs = 250

#this assumes our data format
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#more reshaping
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

#build our model
model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#another one
model.add(Conv2D(32, (3, 3), activation='relu'))
#another one
model.add(Conv2D(32, (3, 3), activation='relu'))
#another one
model.add(Conv2D(32, (3, 3), activation='relu'))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.50))
#another one
model.add(Conv2D(32, (3, 3), activation='relu'))
#another one
model.add(Conv2D(32, (3, 3), activation='relu'))
#another one
model.add(Conv2D(32, (3, 3), activation='relu'))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.50))
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.50))

#flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#fully connected to get all relevant data
model.add(Dense(32, activation='relu'))
#one more dropout for convergence' sake :)
model.add(Dropout(0.25))
#output a softmax to squash the matrix into output probabilities
model.add(Dense(classes, activation='softmax'))
#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#train that ish!
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

 #how well did it do?
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Save the model
# serialize model to JSON
model_json = model.to_json()
with open("PokeModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("PokeModel.h5")
print("Saved model to disk")
