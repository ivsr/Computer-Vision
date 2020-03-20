import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import cv2
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
import os
import random
from tensorflow import  keras
import sys

apple_dir = '/home/ivsr/CV_Group/2_fruits/app_create'
banana_dir = '/home/ivsr/CV_Group/2_fruits/bana_create'
avocado_dir = '/home/ivsr/CV_Group/2_fruits/avocad_create'
mangostan_dir = '/home/ivsr/CV_Group/2_fruits/mastan_create'
orange_dir = '/home/ivsr/CV_Group/2_fruits/oran_create'
guava_dir = '/home/ivsr/CV_Group/2_fruits/guav_create'

#pineapple_dir = '/home/ivsr/CV_Group/2_fruits/pinea_create'


apple_imgs = ['/home/ivsr/CV_Group/2_fruits/app_create/{}'.format(i) for i in os.listdir(apple_dir) if 'apple' in i]
banana_imgs = ['/home/ivsr/CV_Group/2_fruits/bana_create/{}'.format(i) for i in os.listdir(banana_dir) if 'banana' in i]
avocado_imgs = ['/home/ivsr/CV_Group/2_fruits/avocad_create/{}'.format(i) for i in os.listdir(avocado_dir) if 'Avocado' in i]
mangostan_imgs = ['/home/ivsr/CV_Group/2_fruits/mastan_create/{}'.format(i) for i in os.listdir(mangostan_dir) if 'Mangostan' in i]
orange_imgs = ['/home/ivsr/CV_Group/2_fruits/oran_create/{}'.format(i) for i in os.listdir(orange_dir) if 'orange' in i]
guava_imgs = ['/home/ivsr/CV_Group/2_fruits/guav_create/{}'.format(i) for i in os.listdir(guava_dir) if 'Guava' in i]

#pinea_imgs = ['/home/ivsr/CV_Group/2_fruits/pinea_create/{}'.format(i) for i in os.listdir(pineapple_dir) if 'pinea' in i]
print(len(apple_imgs))
print(len(banana_imgs))
print(len(avocado_imgs))
print(len(mangostan_imgs))


#sys.exit()
train_images = apple_imgs + banana_imgs + avocado_imgs + mangostan_imgs + orange_imgs + guava_imgs

fruits = {
    'apple':0,
    'banana':1,
    'Avocado':2,
    'Mangostan':3,
    'orange':4,
    'Guava':5
    #'pinea':4
}
class_name= ['apple', 'banana', 'Avocado', 'Mangostan', 'orange', 'Guava']

#print(train_images.shape)
print(len(train_images))
random.shuffle(train_images)
nrows = 50
ncolumns = 50
channels = 3
def read_and_process_image(list_of_images):
    x=[] #images
    y=[]  #labels
    for image in list_of_images:
        z=cv2.imread(image)
        z=cv2.cvtColor(z,cv2.COLOR_BGR2RGB)
        z=cv2.resize(z,(nrows, ncolumns))
        x.append(z)
        for i,j in fruits.items():
            if i in image:
                y.append(j)
    return x,y
x,y = read_and_process_image(train_images)
print(len(y))
batch_size=32
x= np.array(x)
y=np.array(y)
print(y.shape)
ntrain = len(train_images)
#sys.exit()
#setup the layers
model = keras.Sequential(
    [
        keras.layers.Conv2D(32,(3,3), activation='relu',input_shape=(50, 50, 3)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64,(3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(128,(3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(128,(3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(6, activation='softmax')
    ]
)
model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
#for i in range(2):
train_datagen = ImageDataGenerator(
                                    rescale=1./255.0)
train_generator = train_datagen.flow(x, y, batch_size= batch_size)

history = model.fit_generator(train_generator,
                steps_per_epoch=ntrain//4,
                epochs = 3,
                use_multiprocessing=True,
                workers=8)
#return history
model.save_weights('model_weights_6.h5')
model.save('model_keras_6.h5')
acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.title('Training accurarcy')
plt.legend()
plt.show()
plt.figure()
#Train loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()

