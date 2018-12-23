#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:51:29 2018

@author: yuchunliu
"""

import tensorflow as tf

import time
from datetime import timedelta


from mnist import MNIST
import os
data = MNIST(data_dir="data/MNIST/")

img_size = data.img_size

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = data.num_channels

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Start the network with the noisy input image.
net = x_image



# 1st convolutional layer.
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=32, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# 2nd convolutional layer.
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=64, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# Flatten layer.This should eventually be replaced by:
# net = tf.layers.flatten(net)
net = tf.contrib.layers.flatten(net)

# 1st fully-connected / dense layer.
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=1024, activation=tf.nn.relu)

# 2nd fully-connected / dense layer.
net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)

# Unscaled output of the network.
logits = net

# Softmax output of the network.
y_pred = tf.nn.softmax(logits=logits)

# Loss measure to be optimized.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,
                                                           logits=logits)
loss = tf.reduce_mean(cross_entropy)

gradient = tf.gradients(loss, x)[0]
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)


y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)


        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        
        session.run(optimizer, feed_dict=feed_dict_train)
       

        # Print status every 100 iterations.
        if (i % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
optimize(num_iterations=1500)

saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

saver.save(sess=session, save_path=save_dir)

