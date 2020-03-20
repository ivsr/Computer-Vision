import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import cv2
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
import os
import random
from tensorflow import keras
import sys

import tensorflow as tf
import pickle
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

DATADIR = "/home/ivsr/CV_Group/fruits_tensorflow/train_me"
CATEGORIES = ["apple", "banana_bunch", "mango", "orange"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)  # create path to fruits
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # convert to array
        break
    break

IMG_SIZE = 100


def create_data(data_dir):
    data = []
    for category in CATEGORIES:
        path = os.path.join(data_dir, category)  # create path to fruits
        class_num = CATEGORIES.index(category)  # get the classification
        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                data.append([new_array, class_num])
            except Exception as e:
                pass
    return data


training_data = create_data(DATADIR)
#print(len(training_data))

import random

random.shuffle(training_data)

def get_data(data):
    x = []
    y = []
    for features, label in data:
        x.append(features)
        y.append(label)
    # y = encode(y)
    return x, y
x, y = get_data(training_data)

nrows = 100
ncolums = 100
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.25, random_state=42)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
#print(x_test.shape)
print(y_train[1:10])
#batch_size=32
ntrain = len(x_train)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
#sys.exit()
#setup the layers
model = keras.Sequential(
    [
        keras.layers.Conv2D(16,(5,5), activation='relu',input_shape=(100, 100, 3)),
        keras.layers.MaxPooling2D((2,2)),
        # keras.layers.Conv2D(64,(3,3), activation='relu'),
        # keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(32, (5, 5), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        # keras.layers.Conv2D(128,(3,3), activation='relu'),
        # keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64,(5,5), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ]
)
model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
#for i in range(2):
train_datagen = ImageDataGenerator(rescale=1./255.0)
train_generator = train_datagen.flow(x_train, y_train, batch_size= 32)
x_test = x_test/255.0
#y_test = y_test/255.0

history = model.fit_generator(train_generator,
                steps_per_epoch=ntrain//32,
                epochs = 6,
                use_multiprocessing=True,
                workers=8)
#return history
model.save_weights('model_weights_33.h5')
model.save('model_keras_33.h5')
acc = history.history['acc']
loss = history.history['loss']

results = model.evaluate(x_test, y_test, batch_size=32)
print('test loss, test acc:', results)
#
# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'b', label='Training accurarcy')
# plt.title('Training accurarcy')
# plt.legend()
# plt.show()
# plt.figure()
# #Train loss
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.title('Training loss')
# plt.legend()
# plt.show()
