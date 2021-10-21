# -*-codeing = utf-8 -*-
# @Time : 2021/10/19 19:40
# @Author : Chenyang Wang
# @File : Neural Network.py
# @Software : PyCharm

import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#pysical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0],True)
#Meeting problems while using GPU

(x_train,y_train),(x_test,y_test)=mnist.load_data()
#x_train's shape is（60000，28*28）, there are 60000 data with the resolution 28*28
x_train = x_train.reshape(-1,28*28).astype('float32')/255
x_test = x_test.reshape(-1,28*28).astype('float32')/255
#flatten the pictures and tranform to float32. Normalize the pixels fromm 0-255 to 0-1 to simplify the computation

#both x_train and x_test are numpy array, we should convert them to a tf Tensor
x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

#To build a NN using Sequential API of keras(Comvenient but not flexible) one input==> one output
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512,activation='relu'),
        layers.Dense(256,activation='relu'),
        layers.Dense(10),
    ]
)

#Another way to build NN
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(10))

print(model.summary())  #print the details of model

#Functional API more flexible (like Pytorch)
inputs = keras.Input(shape=(784))
x = layers.Dense(512,activation='relu',name='first_layer')(inputs)
x = layers.Dense(256,activation='relu',name='second_layer')(x)
outputs = layers.Dense(10,activation='Softmax',name='output_layer')(x)
model = keras.Model(inputs=inputs,outputs=outputs)

print(model.summary())  #print the details of model


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)
model.fit(x_train,y_train,verbose=2,epochs=5,batch_size=32)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)