import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import cv2


model = tf.keras.models.load_model('/home/ivsr/CV_Group/2_fruits/model_keras.h5')
model.load_weights('/home/ivsr/CV_Group/2_fruits/model_weights.h5')


test_dir = '/home/ivsr/CV_Group/2_fruits/test'
test_images = ['/home/ivsr/CV_Group/2_fruits/test/{}'.format(i) for i in os.listdir(test_dir)]
a=len(test_images)
print(a)

class_name= ['apple', 'banana',]
t=len(class_name)

def read_and_process_image(list_of_images):
    x=[] #images
    y=[]  #labels
    for i in list_of_images:
        z=cv2.imread(i)
        z=cv2.cvtColor(z,cv2.COLOR_BGR2RGB)
        x.append(z)
        if 'banana' in i:
            y.append(1)
        elif 'apple' in i:
            y.append(0)
    return x,y
x_test, y_test = read_and_process_image(test_images[0:20])
print(y_test)
X = np.array(x_test)
test_datagen = ImageDataGenerator(rescale=1./255)
i = 0
columns=5
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(X, batch_size=1):
    pred = model.predict(batch)
    print(pred)
    if pred > 0.5:
        text_labels.append('a banana ')
    else:
        text_labels.append('an apple')
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 20 == 0:
        break
plt.show()
# x_test, y_test = read_and_process_image(test_images)
# print(y_test)
#
# X = np.array(x_test)
# X= X/255.0
# prediction = model.predict(X)
# print(prediction)
# pred= np.array(prediction)
# print(pred)

# def plot_image(i, predictions_array, true_label, img):
#     prediction_array, true_label, img = predictions_array, true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(img, cmap=plt.cm.binary)
#     predict_label = np.argmax(predictions_array)
#     print(predict_label)
#     print(true_label)
#     print(predictions_array)
#     if predict_label == true_label:
#         color='blue'
#     else:
#         color='red'
#     plt.xlabel("{} {:1.0f}% ({})".format(class_name[predict_label],
#                                         100*np.max(predictions_array),
#                                          class_name[true_label]),
#                                          color='blue')
#
# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array, true_label[i]
#     plt.grid(False)
#     plt.grid(False)
#     plt.xticks(range(t))
#     plt.yticks([])
#     thisplot=plt.bar(range(t), predictions_array, color="#777777")
#     plt.ylim([0,1])
#     predict_label = np.argmax(predictions_array)
#     thisplot[predict_label].set_color('red')
#     thisplot[true_label].set_color('blue')
#
#
# #plot several images with their predictions
# num_rows=5
# num_cols=4
# num_images=num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, pred[i], y_test, X)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, pred[i], y_test)
#     plt.tight_layout()
# plt.show()
