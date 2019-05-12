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

# correct labels
y_ = tf.placeholder(tf.float32, shape=[100, 10])  # YOUR CODE)

# input data
x = tf.placeholder(tf.float32, shape=[100, 784])  # YOUR CODE)

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

hidden_size =  # YOUR CODE
num_classes = 2
W_fc1 =  # YOUR CODE
b_fc1 =  # YOUR CODE
h_fc1 =  # YOUR CODE

# YOUR CODE (how many layers to use, and the number of hidden units)

y = tf.nn.softmax()  # YOUR CODE )

"""Complete the snippet below using your own code.

define the loss function and Optimizer
"""

# define the loss function
cross_entropy =  # YOUR CODE

# define Optimizer
Optimizer = tf.train  # YOUR CODE (cross_entropy)


correct_prediction =  # YOUR CODE
accuracy =  # YOUR CODE
init = tf.global_variables_initializer()

"""The next code snippet trains and evaluates the network. It does this by opening a session to run the tensorflow graph that we have defined.
Complete the code at the locations marked #YOUR CODE below, in order to train the network and to evaluate its accuracy every 50 steps.
"""

with tf.Session() as sess:
    sess.run(init)
for i in range(700):
    input_images, correct_predictions = mnist.train.next_batch(batch_size)

    # YOUR CODE

    sess.run(train_step, feed_dict={   })# YOUR CODE
    if i % 50 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={  # YOUR CODE } )
            print("step %d, training accuracy %g" % (i, train_accuracy))
        # validate

        # YOUR CODE

        test_accuracy = sess.run([accuracy], feed_dict={  # YOUR CODE})
            # YOUR CODE

            print("Validation accuracy: %g." % test_accuracy)
