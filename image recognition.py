# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:02:14 2018

@author: Aymane
"""
123
#getting the data
from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
digit=test_images[3]
#building the model
from keras import models 
from keras import layers 
network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,) ))
network.add(layers.Dense(10,activation='sigmoid'))

#the compilation step
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#prepare thz image data 
train_images=train_images.reshape(60000,28*28)
train_images=train_images.astype('float32')/255

test_images=test_images.reshape(10000,28*28)
test_images=test_images.astype('float32')/255

#preparing the labels
from keras.utils import to_categorical
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#fitting the model
e=5
network.fit(train_images,train_labels,epochs=e,batch_size=128,verbose=0)

#testing the model
test_loss,test_acc=network.evaluate(test_images,test_labels,verbose)
print(test_loss,test_acc)

#displaying the forth digit 
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
