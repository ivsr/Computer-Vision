import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

model = tf.keras.models.load_model('/home/ivsr/CV_Group/2_fruits/model_keras_4.h5')
model.load_weights('/home/ivsr/CV_Group/2_fruits/model_weights_4.h5')


test_dir = '/home/ivsr/CV_Group/2_fruits/test4'
test_images = ['/home/ivsr/CV_Group/2_fruits/test4/{}'.format(i) for i in os.listdir(test_dir)]
a=len(test_images)
print(a)

fruits = {
    'apple':0,
    'banana':1,
    'tomato':2,
    'mangostan':3
}
class_name= ['apple', 'banana', 'tomato','mangostan']
t=len(class_name)
nrows = 50
ncolumns = 50
channels = 3
#print(fruits.items())
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
x_test, y_test = read_and_process_image(test_images[0:a])
print(y_test)
#sys.exit()
X = np.array(x_test)
X= X/255.0
prediction = model.predict(X)
print(prediction)
pred= np.array(prediction)
print(pred)

def plot_image(i, predictions_array, true_label, img):
    prediction_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predict_label = np.argmax(predictions_array)
    print(predict_label)
    print(true_label)
    print(predictions_array)
    if predict_label == true_label:
        color='blue'
    else:
        color='red'
    plt.xlabel("{} {:1.0f}% ({})".format(class_name[predict_label],
                                        100*np.max(predictions_array),
                                         class_name[true_label]),
                                         color='blue')

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.grid(False)
    plt.xticks(range(t))
    plt.yticks([])
    thisplot=plt.bar(range(t), predictions_array, color="#777777")
    plt.ylim([0,1])
    predict_label = np.argmax(predictions_array)
    thisplot[predict_label].set_color('red')
    thisplot[true_label].set_color('blue')


#plot several images with their predictions
num_rows=4
num_cols=3
num_images=num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, pred[i], y_test, X)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, pred[i], y_test)
    plt.tight_layout()
plt.show()

img = x_test[1]
img = (np.expand_dims(img,0))
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(1, predictions_single[0], y_test)
_= plt.xticks(range(10), class_name, rotation=45)