import tensorflow as tf
from time import time

# Download the dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


"""For this, start by defining two placeholders, one to hold the images, and the second to hold the two classes.
Use tf.float32 for the placeholder type.
"""

batch_size = 100
N = batch_size
W = 28
H = 28
D = W * H
num_classes = 10
# [N=100,H,W,C=1]

y = tf.placeholder(tf.float32, shape=[N, num_classes])  # YOUR CODE)

# input data
x = tf.placeholder(tf.float32, shape=[N, D])  # YOUR CODE)

"""Next, define the network itself. It is up to you how many layers to use, and the number of hidden units in each layer.

You are allowed to use only the following functions:
* weight_variable
* bias_variable
* tf.nn.relu
* tf.nn.softmax
* tf.matmul

Please note that each layer includes not only tf.matmul, but also a bias variable.
"""

# build the net

# hidden_size =  # YOUR CODE
#num_classes = 2
W_fc1 = weight_variable([D, num_classes])
b_fc1 = bias_variable([num_classes])
h_fc1 = tf.nn.relu(tf.add(tf.matmul(x, W_fc1), b_fc1)) # YOUR CODE

W_fc2 = weight_variable([num_classes, D]) #10x784
b_fc2 = bias_variable([D])
h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2)) # 100x784

W_fc3 = weight_variable([D, num_classes])
b_fc3 = bias_variable([num_classes])
logits = tf.nn.relu(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

hypothesis = tf.nn.softmax(logits)

"""Complete the snippet below using your own code.

define the loss function and Optimizer
"""

#define the loss function
cross_entropy =  # YOUR CODE

# define Optimizer
Optimizer = tf.train  # YOUR CODE (cross_entropy)


correct_prediction =  tf.placeholder # YOUR CODE
accuracy =  #true_prediction/N (num of samples in Batch Size
init = tf.global_variables_initializer()

"""The next code snippet trains and evaluates the network. It does this by opening a session to run the tensorflow graph that we have defined.
Complete the code at the locations marked #YOUR CODE below, in order to train the network and to evaluate its accuracy every 50 steps.
"""

with tf.Session() as sess:
    sess.run(init)
for i in range(700):
    input_images, correct_predictions = mnist.train.next_batch(batch_size)

    # YOUR CODE
    train_step = tf.Variable()
    sess.run(train_step, feed_dict={x: input_images, y:correct_predictions})# YOUR CODE
    if i % 50 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={  # YOUR CODE } )
        print("step %d, training accuracy %g" % (i, train_accuracy))
        # validate

        # YOUR CODE
        test_images, test_predictions = mnist.test.next_batch(batch_size)
        test_accuracy = sess.run([accuracy], feed_dict={  # YOUR CODE})
            # YOUR CODE

        print("Validation accuracy: %g." % test_accuracy)
