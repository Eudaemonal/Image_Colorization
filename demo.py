import tensorflow as tf  # 0.12  
import numpy as np
import os
import glob
import sys
from matplotlib import pyplot as plt


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


def color_net():
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
        conv2 = conv2d(conv1, weights['wc2'], sigmoid=False, bn=True)
        # upscale to 112x112x128  
        conv2 = tf.image.resize_bilinear(conv2, (112, 112))
        conv2 = tf.add(conv2, batch_norm(conv2_2, 128, phase_train))

        # Bx112x112x128 -> 3x3 conv = Bx112x112x64  
        conv3 = conv2d(conv2, weights['wc3'], sigmoid=False, bn=True)
        # upscale to Bx224x224x64  
        conv3 = tf.image.resize_bilinear(conv3, (224, 224))
        conv3 = tf.add(conv3, batch_norm(conv1_2, 64, phase_train))

        # Bx224x224x64 -> 3x3 conv = Bx224x224x3  
        conv4 = conv2d(conv3, weights['wc4'], sigmoid=False, bn=True)
        conv4 = tf.add(conv4, batch_norm(grayscale, 3, phase_train))

        # Bx224x224x3 -> 3x3 conv = Bx224x224x3  
        conv5 = conv2d(conv4, weights['wc5'], sigmoid=False, bn=True)
        # Bx224x224x3 -> 3x3 conv = Bx224x224x2  
        conv6 = conv2d(conv5, weights['wc6'], sigmoid=True, bn=True)

    return conv6

def train_color_net():
    pred = color_net()
    pred_yuv = tf.concat([tf.split(grayscale_yuv, 3, 3)[0], pred],3)
    pred_rgb = yuv2rgb(pred_yuv)

    colorimage_yuv = rgb2yuv(colorimage)
    loss = tf.square(tf.subtract(pred, tf.concat([tf.split(colorimage_yuv, 3, 3)[1], tf.split(colorimage_yuv, 3, 3)[2]],3)))

    if uv == 1:
        loss = tf.split(loss, 2,3)[0]
    elif uv == 2:
        loss = tf.split(loss, 2, 3)[1]
    else:
        loss = (tf.split(loss,2, 3 )[0] + tf.split(loss,2, 3 )[1]) / 2

    global_step = tf.Variable(0, name='global_step', trainable=False)
    if phase_train is not None:
        optimizer = tf.train.GradientDescentOptimizer(0.0001)
        opt = optimizer.minimize(loss, global_step=global_step, gate_gradients=optimizer.GATE_NONE)

        # Saver.
    saver = tf.train.Saver()
    sess = tf.Session()
    # Initialize the variables.  
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    # Start input enqueue threads.  
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps  
            training_opt = sess.run(opt, feed_dict={phase_train: True, uv: 1})
            training_opt = sess.run(opt, feed_dict={phase_train: True, uv: 2})

            step = sess.run(global_step)

            if step % 1 == 0:
                pred_, pred_rgb_, colorimage_, grayscale_rgb_, cost = sess.run(
                    [pred, pred_rgb, colorimage, grayscale_rgb, loss], feed_dict={phase_train: False, uv: 3})
                print({"step": step, "cost": np.mean(cost)})
                if step % 10 == 0:
                    summary_image = concat_images(grayscale_rgb_[0], pred_rgb_[0])
                    summary_image = concat_images(summary_image, colorimage_[0])
                    plt.imsave("summary/" + str(step) + "_0.jpg", summary_image)

            if step % 100000 == 400:
                if not os.path.exists(checkpoints_dir):
                    os.makedirs(checkpoints_dir)
                save_path = saver.save(sess, checkpoints_dir + "/color_net_model"+str(step)+".ckpt")
                print("Model saved in file: %s" % save_path)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.  
        coord.request_stop()

        # Wait for threads to finish.
    coord.join(threads)
    sess.close()



if __name__ == "__main__":

    img = cv2.imread('gray.jpg')
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    with open("vgg16-20160129.tfmodel", mode='rb') as f:
        fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)