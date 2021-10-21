# -*-codeing = utf-8 -*-
# @Time : 2021/10/20 12:02
# @Author : Chenyang Wang
# @File : RNN.py
# @Software : PyCharm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#RNN
model = keras.Sequential()
model.add(keras.Input(shape=(None,28))) #don't need to specify the time step actually we have 28 time step
model.add(
    layers.SimpleRNN(256,return_sequences=True,activation='relu')
)
model.add(
    layers.SimpleRNN(256,activation='relu') #default activation function: tanh
)
model.add(layers.Dense(10))

#GRU
model = keras.Sequential()
model.add(keras.Input(shape=(None,28))) #don't need to specify the time step actually we have 28 time step
model.add(
    layers.GRU(256,return_sequences=True,activation='relu')
)
model.add(
    layers.GRU(256,activation='relu') #default activation function: tanh
)
model.add(layers.Dense(10))
print(model.summary())

#LSTM
model = keras.Sequential()
model.add(keras.Input(shape=(None,28))) #don't need to specify the time step actually we have 28 time step
model.add(
    layers.LSTM(256,return_sequences=True,activation='relu')
)
model.add(
    layers.LSTM(256,activation='relu') #default activation function: tanh
)
model.add(layers.Dense(10))

#Bidirectional LSTM
model = keras.Sequential()
model.add(keras.Input(shape=(None,28))) #don't need to specify the time step actually we have 28 time step
model.add(
    layers.Bidirectional(layers.LSTM(256,return_sequences=True,activation='relu'))
)
model.add(
    layers.Bidirectional(layers.LSTM(256,activation='relu')) #default activation function: tanh
)
model.add(layers.Dense(10))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=2)
model.evaluate(x_test,y_test,batch_size=64,epochs=10,verbose=2)