# -*-codeing = utf-8 -*-
# @Time : 2021/10/20 11:22
# @Author : Chenyang Wang
# @File : CNN.py
# @Software : PyCharm
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
#transform to float32. Normalize the pixels fromm 0-255 to 0-1 to simplify the computation
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#Using Sequential
model = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)), #using convolution, 32*32 in 3 channel(rgb)
        layers.Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu'), #padding='same' keep the input size same
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'),
        layers.Flatten(),
        layers.Dense(100,activation='relu'),
        layers.Dense(10)
    ]
)

#Using functional API
def my_model():
    inputs = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(
        32,3,padding='same',kernel_regularizer=regularizers.l2(0.01)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        64,5,padding='same',kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activation.relu(x)
    x = layers.Conv2D(
        128,3,padding='same',kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activation.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100,activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs,outputs=outputs)
    return model

model = my_model()

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), #from_logits means loss includes Softmax
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"],
)
model.fit(x_train,y_train,batch_size=64,epochs=150,verbose=2)