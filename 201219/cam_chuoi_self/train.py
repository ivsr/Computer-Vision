import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import cv2
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import os
import random
from tensorflow import  keras
import sys

#train_images, train_set_y, test_images, test_set_y, classes = load_dataset()
orange_dir = '/home/ivsr/CV_Group/cam_chuoi_self/orang'
banana_bunch_dir = '/home/ivsr/CV_Group/cam_chuoi_self/bana_bunch'
orange_imgs = ['/home/ivsr/CV_Group/cam_chuoi_self/orang/{}'.format(i) for i in os.listdir(orange_dir) if 'orange' in i]
banana_bunch_imgs = ['/home/ivsr/CV_Group/cam_chuoi_self/bana_bunch/{}'.format(i) for i in os.listdir(banana_bunch_dir) if 'banana_bunch' in i]
#tomato_imgs = ['/home/ivsr/CV_Group/2_fruits/tom_create/{}'.format(i) for i in os.listdir(banana_bunch_dir) if 'tomato' in i]
test_dir = '/home/ivsr/CV_Group/cam_chuoi_self/test'
test_imgs =['/home/ivsr/CV_Group/cam_chuoi_self/test/{}'.format(i) for i in os.listdir(test_dir)]
train_images = orange_imgs + banana_bunch_imgs

print(len(test_imgs))
#sys.exit()
fruits = {
    'orange':0,
    'banana_bunch':1,
}
class_name= ['orange', 'banana_bunch']

#print(train_images.shape)
print(len(train_images))
random.shuffle(train_images)
nrows = 100
ncolumns = 100
channels = 3
def read_and_process_image(list_of_images):
    x=[] #images
    y=[]  #labels
    for i in list_of_images:
        z=cv2.imread(i)
        z=cv2.cvtColor(z,cv2.COLOR_BGR2RGB)
        z=cv2.resize(z,(nrows, ncolumns))
        x.append(z)
        if 'banana_bunch' in i:
            y.append(1)
        elif 'orange' in i:
            y.append(0)
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
        keras.layers.Conv2D(32,(3,3), activation='relu',input_shape=(100, 100, 3)),
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
        keras.layers.Dense(1, activation='sigmoid')
    ]
)
rms= keras.optimizers.RMSprop(lr=1e-4)
model.compile(optimizer= rms, loss='binary_crossentropy', metrics=['acc'])
#for i in range(2):
train_datagen = ImageDataGenerator(
                                    rescale=1./255.0)
                                    #rotation_range=40)
                                   #  width_shift_range=0,
                                   #  height_shift_range=0.2,
                                   # # brightness_range= [0.4,1.6],t
                                   #  shear_range=0.2,
                                   #  zoom_range=0.3,
                                    #horizontal_flip=True,)
train_generator = train_datagen.flow(x, y, batch_size= batch_size)

history = model.fit_generator(train_generator,
                steps_per_epoch=ntrain//16,
                epochs = 10,
                use_multiprocessing=True,
                workers=8)
#return history
model.save_weights('model_weights.h5')
model.save('model_keras.h5')
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

x_test, y_test = read_and_process_image(test_imgs[0:12])
print(y_test)
X = np.array(x_test)
test_datagen = ImageDataGenerator(rescale=1./255)
i = 0
columns=4
rows = 3
text_labels = []
plt.figure(figsize=(30,20))
for batch in test_datagen.flow(X, batch_size=1):
    pred = model.predict(batch)
    print(pred)
    if pred > 0.5:
        text_labels.append('a banana_bunch ')
    else:
        text_labels.append('an orange')
    plt.subplot(rows, 2*columns, 2*i+1)
    plt.title('This is ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 12 == 0:
        break
plt.show()