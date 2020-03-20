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
#train_images, train_set_y, test_images, test_set_y, classes = load_dataset()
apple_dir = '/home/ivsr/CV_Group/2_fruits/app_create'
banana_dir = '/home/ivsr/CV_Group/2_fruits/bana_create'
apple_imgs = ['/home/ivsr/CV_Group/2_fruits/app_create/{}'.format(i) for i in os.listdir(apple_dir) if 'apple' in i]
banana_imgs = ['/home/ivsr/CV_Group/2_fruits/bana_create/{}'.format(i) for i in os.listdir(banana_dir) if 'banana' in i]
tomato_imgs = ['/home/ivsr/CV_Group/2_fruits/tom_create/{}'.format(i) for i in os.listdir(banana_dir) if 'tomato' in i]

train_images = apple_imgs + banana_imgs

fruits = {
    'apple':0,
    'banana':1,
}
class_name= ['apple', 'banana']

#print(train_images.shape)
print(len(train_images))
random.shuffle(train_images)
nrows = 50
ncolumns = 50
channels = 3
def read_and_process_image(list_of_images):
    x=[] #images
    y=[]  #labels
    for i in list_of_images:
        z=cv2.imread(i)
        z=cv2.cvtColor(z,cv2.COLOR_BGR2RGB)
        z=cv2.resize(z,(nrows, ncolumns))
        x.append(z)
        if 'banana' in i:
            y.append(1)
        elif 'apple' in i:
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
                steps_per_epoch=ntrain//32,
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

x_test, y_test = read_and_process_image(test_images[0:5])
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
    if i % 5 == 0:
        break
plt.show()
