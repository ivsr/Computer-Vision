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
#%matplotlib inline

#train_images, train_set_y, test_images, test_set_y, classes = load_dataset()
apple_dir = '/home/ivsr/CV_Group/2_fruits/app_create'
banana_dir = '/home/ivsr/CV_Group/2_fruits/bana_create'
apple_imgs = ['/home/ivsr/CV_Group/2_fruits/app_create/{}'.format(i) for i in os.listdir(apple_dir)]
banana_imgs = ['/home/ivsr/CV_Group/2_fruits/bana_create/{}'.format(i) for i in os.listdir(banana_dir)]
train_images = apple_imgs + banana_imgs
test_dir = '/home/ivsr/CV_Group/2_fruits/test'
test_images = ['/home/ivsr/CV_Group/2_fruits/test/{}'.format(i) for i in os.listdir(test_dir)]
print(len(test_images))
fruits = {
    'apple':0,
    'banana':1,
}
class_name= ['apple', 'banana']

#print(train_images.shape)
print(len(train_images))
random.shuffle(train_images)

def read_and_process_image(list_of_images):
    x=[] #images
    y=[]  #labels
    for i in list_of_images:
        z=cv2.imread(i)
#        z = img_to_array(z)
        x.append(z)
        if 'banana' in i:
            y.append(1)
        elif 'apple' in i:
            y.append(0)
    return x,y
train_set_x,train_set_y = read_and_process_image(train_images)
test_set_x, test_set_y = read_and_process_image(test_images)
print(len(test_set_x))
# Example of a picture
train_set_x = np.array(train_set_x)
train_set_y = np.array(train_set_y)
test_set_x = np.array(test_set_x)
test_set_y = np.array(test_set_y)
index = 25

m_train = train_set_x.shape[0]
m_test = test_set_x.shape[0]
num_px = train_set_x.shape[1]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x.shape))
print ("test_set_x shape: " + str(test_set_x.shape))

X=train_set_x
Y=test_set_x
train_set_x_flatten = X.reshape(X.shape[0], -1).T
test_set_x_flatten = Y.reshape(Y.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    ### END CODE HERE ###

    return s
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))

# GRADED FUNCTION: propagate

def propagate(w, b, X, Y):
    m = X.shape[1]
    A1=np.dot(w.T,X)+b
    A = sigmoid(A1)
    x=np.dot(Y,np.log(A))
    x1=np.dot((1-Y),np.log(1-A))
    cost = (-1/m)*(np.sum(x+x1))                    # compute cost
    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1/m*np.sum(A-Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}
    return grads, cost
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b- learning_rate*db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

# GRADED FUNCTION: predict

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A1=np.dot(w.T,X)+b
    A = sigmoid(A1)
    for i in range(A.shape[1]):
        Y_prediction = A >0.5
        #Y_prediction[0,i]= A[0,i] >0.5
        pass

    assert(Y_prediction.shape == (1, m))

    return Y_prediction

# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Example of a picture that was wrongly classified.
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + class_name[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "my_image.jpg"   # change this to the name of your image file
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
image = image/255.
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + class_name[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")



