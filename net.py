import tensorflow as tf  # 0.13 
import numpy as np
import os
import glob
import sys
from matplotlib import pyplot as plt
from utils import batch_norm,conv2d


def color_net(graph,phase_train, grayscale):
    with tf.variable_scope('vgg'):
        conv1_2 = graph.get_tensor_by_name("import/conv1_2/Relu:0")
        conv2_2 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
        conv3_3 = graph.get_tensor_by_name("import/conv3_3/Relu:0")
        conv4_3 = graph.get_tensor_by_name("import/conv4_3/Relu:0")

        # Store layers weight
    weights = {
        # 1x1 conv, 512 inputs, 256 outputs  
        'wc1': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01)),
        # 3x3 conv, 512 inputs, 128 outputs  
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01)),
        # 3x3 conv, 256 inputs, 64 outputs  
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01)),
        # 3x3 conv, 128 inputs, 3 outputs  
        'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01)),
        # 3x3 conv, 6 inputs, 3 outputs  
        'wc5': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01)),
        # 3x3 conv, 3 inputs, 2 outputs  
        'wc6': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01)),
    }


    with tf.variable_scope('color_net'):
        # Bx28x28x512 -> batch norm -> 1x1 conv = Bx28x28x256  
        conv1 = tf.nn.relu(tf.nn.conv2d(batch_norm(conv4_3, 512, phase_train), weights['wc1'], [1, 1, 1, 1], 'SAME'))
        # upscale to 56x56x256  
        conv1 = tf.image.resize_bilinear(conv1, (56, 56))
        conv1 = tf.add(conv1, batch_norm(conv3_3, 256, phase_train))

        # Bx56x56x256-> 3x3 conv = Bx56x56x128  
        conv2 = conv2d(conv1, weights['wc2'], phase_train,sigmoid=False, bn=True)
        # upscale to 112x112x128  
        conv2 = tf.image.resize_bilinear(conv2, (112, 112))
        conv2 = tf.add(conv2, batch_norm(conv2_2, 128, phase_train))

        # Bx112x112x128 -> 3x3 conv = Bx112x112x64  
        conv3 = conv2d(conv2, weights['wc3'],phase_train, sigmoid=False, bn=True)
        # upscale to Bx224x224x64  
        conv3 = tf.image.resize_bilinear(conv3, (224, 224))
        conv3 = tf.add(conv3, batch_norm(conv1_2, 64, phase_train))

        # Bx224x224x64 -> 3x3 conv = Bx224x224x3  
        conv4 = conv2d(conv3, weights['wc4'], phase_train,sigmoid=False, bn=True)
        conv4 = tf.add(conv4, batch_norm(grayscale, 3, phase_train))

        # Bx224x224x3 -> 3x3 conv = Bx224x224x3  
        conv5 = conv2d(conv4, weights['wc5'],phase_train, sigmoid=False, bn=True)
        # Bx224x224x3 -> 3x3 conv = Bx224x224x2  
        conv6 = conv2d(conv5, weights['wc6'],phase_train, sigmoid=True, bn=True)

    return conv6
