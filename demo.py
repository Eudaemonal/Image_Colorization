import tensorflow as tf  # 0.13 
import numpy as np
import os
import glob
import sys
from matplotlib import pyplot as plt
from utils import rgb2yuv,yuv2rgb,input_pipeline,concat_images
from net import color_net



if __name__ == "__main__":
    filenames = glob.glob("demo/*")
    with open("vgg16-20160129.tfmodel", mode='rb') as f:
        fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    batch_size = 4
    num_epochs = 1e+9
    colorimage = input_pipeline(filenames, batch_size, num_epochs=num_epochs)
    grayscale = tf.image.rgb_to_grayscale(colorimage)
    grayscale_rgb = tf.image.grayscale_to_rgb(grayscale)
    grayscale_yuv = rgb2yuv(grayscale_rgb)
    grayscale = tf.concat([grayscale, grayscale, grayscale],3)

    
    tf.import_graph_def(graph_def, input_map={"images": grayscale})
    graph = tf.get_default_graph()

   
    # Saver.
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('checkpoints/color_net_model200500.ckpt.meta')
        saver.restore(sess, "checkpoints/color_net_model200500.ckpt")

        pred_rgb_ = tf.placeholder(dtype=tf.float32)
        pred_rgb_ = sess.run(["pred_rgb:0"], feed_dict={"phase_train:0": False, "uv:0": 3})
        

        plt.imsave("demo/color.jpg", pred_rgb_[0])

    sess.close()