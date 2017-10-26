import tensorflow as tf
import cv2
from utils import *
from net import Net
from skimage.io import imsave
import numpy as np



filename = 'demo/gray/1.jpg'
img = cv2.imread(filename)

if len(img.shape) == 3:
   img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


img = img[None, :, :, None]
data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 -50

autocolor = Net(train=False)
conv8_313 = autocolor.inference(data_l)

saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, 'checkpoints/model.ckpt')
  conv8_313 = sess.run(conv8_313)

img_rgb = decode(data_l, conv8_313,2.63)
imsave('demo/color/1.jpg', img_rgb)
