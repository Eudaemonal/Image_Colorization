import tensorflow as tf  # 0.13 
import numpy as np
import os
import glob
import sys
from matplotlib import pyplot as plt
from utils import rgb2yuv,yuv2rgb,input_pipeline,concat_images
from net import color_net



# train neural networks
def train_color_net(graph, phase_train, uv, grayscale):
    pred_rgb = tf.placeholder(tf.float32, name="pred_rgb")
    pred = color_net(graph, phase_train, grayscale)
    pred_yuv = tf.concat([tf.split(grayscale_yuv, 3, 3)[0], pred],3)
    pred_rgb = yuv2rgb(pred_yuv)

    colorimage_yuv = rgb2yuv(colorimage)
    loss = tf.square(tf.subtract(pred, tf.concat([tf.split(colorimage_yuv, 3, 3)[1], tf.split(colorimage_yuv, 3, 3)[2]],3)))

    if uv == 1:
        loss = tf.split(loss, 2,3)[0]
    elif uv == 2:
        loss = tf.split(loss, 2, 3)[1]
    else:
        loss = (tf.split(loss,2, 3)[0] + tf.split(loss,2, 3)[1]) / 2

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

            if step % 100000 == 500:
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
    # image sources
    filenames = glob.glob("resized/*")

    with open("model/vgg16-20160129.tfmodel", mode='rb') as f:
        fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    if not os.path.exists('summary'):
        os.mkdir('summary')

    batch_size = 4
    num_epochs = 1e+9
    colorimage = input_pipeline(filenames, batch_size, num_epochs=num_epochs)
    grayscale = tf.image.rgb_to_grayscale(colorimage)
    grayscale_rgb = tf.image.grayscale_to_rgb(grayscale)
    grayscale_yuv = rgb2yuv(grayscale_rgb)
    grayscale = tf.concat([grayscale, grayscale, grayscale],3)

    tf.import_graph_def(graph_def, input_map={"images": grayscale})
    graph = tf.get_default_graph()
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    uv = tf.placeholder(tf.uint8, name='uv')

    checkpoints_dir = "./checkpoints"

    train_color_net(graph, phase_train, uv, grayscale)

