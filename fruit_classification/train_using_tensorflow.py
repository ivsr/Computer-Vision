import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys
DATADIR = "/home/ivsr/CV_Group/fruits_tensorflow/train_me"
CATEGORIES = ["apple", "banana_bunch", "mango", "orange"]
#CATEGORIES = [ "banana_bunch", "mango", "orange"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)  # create path to fruits
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB) # convert to array
        break
    break

IMG_SIZE = 50

def create_data(data_dir):
    data = []
    for category in CATEGORIES:
        path = os.path.join(data_dir,category)  # create path to fruits
        class_num = CATEGORIES.index(category)  # get the classification
        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.COLOR_BGR2RGB)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                data.append([new_array, class_num])
            except Exception as e:
                pass
    return data

training_data = create_data(DATADIR)
print(len(training_data))


def encode(data):
    encoded = tf.keras.utils.to_categorical(data)
    return encoded

# print(len(test_data))
import random
random.shuffle(training_data)
# for sample in training_data[:10]:
#     print(sample[1])
def get_data(data):
    x= []
    y= []
    for features,label in data:
        x.append(features)
        y.append(label)
    y = encode(y)
    return x,y
#create train dataset
x,y = get_data(training_data)
#x_train, y_train = get_data(training_data)
#x_test, y_test = get_data(test_data)
(x_train, x_test, y_train, y_test) = train_test_split(x,y, test_size=0.25, random_state=42)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
#sys.exit()
import pickle
#save data : images, labels one hot
pickle_out = open("X_me.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_me.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

#load data to train
pickle_in = open("X_me.pickle","rb")
x_train = pickle.load(pickle_in)

pickle_in = open("y_me.pickle","rb")
y_train = pickle.load(pickle_in)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape)
print (y_train.shape)
print(x_test.shape)
print(y_test.shape)
#sys.exit()

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

x = tf.placeholder(tf.float32, shape=[None,50,50,3], name = "Input")
y_ = tf.placeholder(tf.float32, shape=[None,4],name = "Output")
x_image = tf.reshape(x,[-1,50,50,3])

keep_prob = tf.placeholder(tf.float32)
conv1 = conv_layer(x_image, shape=[5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)
conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)
conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 128])
conv3_pool = max_pool_2x2(conv3)
conv3_flat = tf.reshape(conv3_pool, [-1, 7* 7 * 128])
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)
full_1 = tf.nn.relu(full_layer(conv3_drop, 512))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
y_conv = full_layer(full1_drop, 4)

num_steps = 1000
minibatch_size = 100
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

x_train = x_train/255.0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps+1):
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:x_train,
                                                           y_:y_train, keep_prob: 1.0})
            print('step: {},training accuracy: {}'.format(i,train_accuracy))
        sess.run(train_step, feed_dict={x:x_train, y_:y_train, keep_prob:0.5})
        saver = tf.train.Saver()
        path="/home/ivsr/CV_Group/fruits_tensorflow/"

        saver.save(sess,path + 'my_model',global_step=800)
    test_accuracy = np.mean([sess.run(accuracy,
                            feed_dict={x:x_test, y_:y_test, keep_prob:1.0})])
    print("test accuracy:{}".format((test_accuracy)))
