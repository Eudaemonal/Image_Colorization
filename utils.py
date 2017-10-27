import tensorflow as tf  # 0.13 
import numpy as np
import os
import glob
import sys
from matplotlib import pyplot as plt
from normalize import ConvolutionalBatchNormalizer


def rgb2yuv(rgb):
    """ 
    Convert RGB image into YUV 
    """
    rgb2yuv_filter = tf.constant([[[[0.299, -0.169, 0.499],
                                    [0.587, -0.331, -0.418],
                                    [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    return temp


def yuv2rgb(yuv):
    """ 
    Convert YUV image into RGB
    """
    yuv = tf.multiply(yuv, 255)
    yuv2rgb_filter = tf.constant([[[[1., 1., 1.],
                                    [0., -0.34413999, 1.77199996],
                                    [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp, tf.multiply(tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp


def concat_images(imga, imgb):
    """ 
    Combines two color image ndarrays side-by-side. 
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def read_my_file_format(filename_queue, randomize=False):
    reader = tf.WholeFileReader()
    key, file = reader.read(filename_queue)
    uint8image = tf.image.decode_jpeg(file, channels=3)
    uint8image = tf.random_crop(uint8image, (224, 224, 3))
    if randomize:
        uint8image = tf.image.random_flip_left_right(uint8image)
        uint8image = tf.image.random_flip_up_down(uint8image, seed=None)
    float_image = tf.div(tf.cast(uint8image, tf.float32), 255)
    return float_image


def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)
    example = read_my_file_format(filename_queue, randomize=False)
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.shuffle_batch([example], batch_size=batch_size, capacity=capacity,
                                           min_after_dequeue=min_after_dequeue)
    return example_batch



def batch_norm(x, depth, phase_train):
    with tf.variable_scope('batchnorm'):
        ewma = tf.train.ExponentialMovingAverage(decay=0.9999)
        bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)
        update_assignments = bn.get_assigner()
        x = bn.normalize(x, train=phase_train)
    return x



def conv2d(_X, w, phase_train, sigmoid=False, bn=False):
    with tf.variable_scope('conv2d'):
        _X = tf.nn.conv2d(_X, w, [1, 1, 1, 1], 'SAME')
        if bn:
            _X = batch_norm(_X, w.get_shape()[3], phase_train)
        if sigmoid:
            return tf.sigmoid(_X)
        else:
            _X = tf.nn.relu(_X)
            return tf.maximum(0.01 * _X, _X)
