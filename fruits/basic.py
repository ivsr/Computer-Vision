import tensorflow as tf
import numpy as np
#with tf.compat.v2.device("/device:CPU:0"):
    # a=tf.add(6,7)
    # b=tf.multiply(9,9)
    # print(b.numpy())
    # print(a.numpy())
    # x=tf.constant(1,shape=[5,6])
    # y= tf.zeros(shape=[8,8], dtype= tf.float16)
    # z=tf.ones(shape=[5,5], dtype= tf.float16)
w=tf.fill([1,3],2.)
print(w.numpy())
"""print(y.numpy())
    print(z.numpy())
    print(x.numpy())"""
def sigmoid(x):
    x1=tf.compat.v2.exp(-x)
    sig=1/(1+x1)
    return sig
    #w1=np.array([1,2,3])
t=sigmoid(w)
x=tf.constant(1.,shape=[5,6])
print(sigmoid(x))
print(t.numpy())
