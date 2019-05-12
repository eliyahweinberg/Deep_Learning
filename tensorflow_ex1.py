import tensorflow as tf
import math

"""Repeat the same exercise as above -- calculating the area of a circle using tensorflow, but this time use tf.placeholder to hold the radius value (instead of using a variable)."""

tf.reset_default_graph()
pi = tf.constant(math.pi, tf.float32)
r = tf.placeholder(tf.float32, shape=[1])
y = pi * r * r
with tf.Session() as sess:
    res = sess.run(y, {r: [2]})
    print(res)
