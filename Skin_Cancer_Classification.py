# -*-codeing = utf-8 -*-
# @Time : 2021/10/21 11:58
# @Author : Chenyang Wang
# @File : Skin_Cancer_Classification.py
# @Software : PyCharm

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import random

seed = 1
random.seed(seed)
directory = "ISIC/images/"
train = "data/train/"
test = "data/test/"
validation = "data/validation/"

os.makedirs(train + "benign/")
os.makedirs(train + "malignant/")
os.makedirs(test + "benign/")
os.makedirs(test + "malignant/")
os.makedirs(validation + "benign/")
os.makedirs(validation + "malignant/")

test_examples = train_examples = validation_examples = 0

for line in open("ISIC/labels.csv").readlines()[1:]:
    split_line = line.split(",")
    img_file = split_line[0]
    benign_malign = split_line[1]

    random_num = random.random()

    if random_num < 0.8:
        location = train
        train_examples += 1

    elif random_num < 0.9:
        location = validation
        validation_examples += 1

    else:
        location = test
        test_examples += 1

    if int(float(benign_malign)) == 0:
        shutil.copy(
            "ISIC/images/" + img_file + ".jpg",
            location + "benign/" + img_file + ".jpg",
        )

    elif int(float(benign_malign)) == 1:
        shutil.copy(
            "ISIC/images/" + img_file + ".jpg",
            location + "malignant/" + img_file + ".jpg",
        )

print(f"Number of training examples {train_examples}")
print(f"Number of test examples {test_examples}")
print(f"Number of validation examples {validation_examples}")


train_examples = 20225
test_examples = 2551
validation_examples = 2555
img_height = img_width = 224
batch_size = 32

# NasNet
# model = keras.Sequential([
#    hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
#                   trainable=True),
#    layers.Dense(1, activation="sigmoid"),
# ])

model = keras.models.load_model("isic_model/")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    dtype=tf.float32,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)

train_gen = train_datagen.flow_from_directory(
    "data/train/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

validation_gen = validation_datagen.flow_from_directory(
    "data/validation/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

test_gen = test_datagen.flow_from_directory(
    "data/test/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
]

model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4),
    loss=[keras.losses.BinaryCrossentropy(from_logits=False)],
    metrics=METRICS,
)

model.fit(
    train_gen,
    epochs=1,
    verbose=2,
    steps_per_epoch=train_examples // batch_size,
    validation_data=validation_gen,
    validation_steps=validation_examples // batch_size,
    callbacks=[keras.callbacks.ModelCheckpoint("isic_model")],
)


def plot_roc(labels, data):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp)
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.show()


test_labels = np.array([])
num_batches = 0

for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches += 1
    if num_batches == math.ceil(test_examples / batch_size):
        break

plot_roc(test_labels, test_gen)
model.evaluate(validation_gen, verbose=2)
model.evaluate(test_gen, verbose=2)