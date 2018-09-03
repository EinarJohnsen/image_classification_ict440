import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np


def one_hot_encoder(value):
    answer = [0,0,0,0]
    answer[value] = 1
    return answer

def one_hot_encoder2(value):
    answer = [0,0,0]
    answer[value] = 1
    return answer

def one_hot_encoder3(value):
    answer = [0,0]
    answer[value] = 1
    return answer


def vectorized_result(j):
    e = np.zeros((4, 1))
    e[j] = 1.0
    return e


def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    #image_resized = tf.image.resize_image_with_crop_or_pad(image, 128, 128)
    #final_image = tf.image.per_image_standardization(image)
    return image, label


# HELPER FUNCTIONS


# INIT weights
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return (tf.Variable(init_random_dist))


# INIT Bias
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


# CONV2D
def conv2d(x, W):
    # x --> input tensor [batch, H, W, Channels]
    # W --> [filter H, filter W, Channels IN, Channels OUT]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Pooling
def max_pooling_2by2(x):
    # x --> [batch, h, w, c]
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#Convolutional layer
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    return tf.nn.relu(conv2d(input_x, W) + b)


# Normal (FULLY CONNTCTED)
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    
    return tf.matmul(input_layer, W) + b
