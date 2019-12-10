import numpy as np
import cv2
import matplotlib.image as mpimg
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

anh = '/home/ivsr/CV_Group/2_fruits/Guava'
anh_yeu = ['/home/ivsr/CV_Group/2_fruits/Guava/{}'.format(i) for i in os.listdir(anh)]
print(len(anh_yeu))
a=0
for i in (anh_yeu):
    a+=1
    img = load_img(i)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    datagen = ImageDataGenerator(
                                  # rescale = 1./255,
                                  #rotation_range= 90,
                                  brightness_range=[0.25, 1.75],
                                  #zoom_range = [1.2, 1.8]
                                  #zoom_range = 1.5
                                )

    image = datagen.flow(img, save_to_dir='guav_create', save_prefix='Guava'+str(a), save_format='jpg')
    #print(len(img))
    #sys.exit()
    t= 0
    for x in image:
        t +=1
        if t == 10:
            break
