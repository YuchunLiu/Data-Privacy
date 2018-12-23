#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:09:40 2018

@author: yuchunliu
"""

import tensorflow as tf
import cv2 as cv

from mnist import MNIST
import os

import numpy as np

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

net = x_image

net = tf.layers.conv2d(inputs=net, name='layer_conv0', padding='same',
                       filters=16, kernel_size=3, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=32, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=64, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='same',
                       filters=64, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.contrib.layers.flatten(net)

net = tf.layers.batch_normalization(inputs= net, name = 'batch')

net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=1024, activation=tf.nn.relu)

net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)

logits = net

y_pred = tf.nn.softmax(logits=logits)

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

saver = tf.train.Saver()

batch_size = 64

dir_path = os.listdir('./data/ADV')
'''
adv_imgs = []
for i in range(len(dir_path)):
    img = cv.imread('./data/ADV/'+dir_path[i],0)
    img_flat =img.reshape(-1,784)[0]/255.0
    adv_imgs.append(img_flat)
    
adv_imgs = np.asarray(adv_imgs)
'''
adv_imgs =np.load('adv.npy')
adv_labels = np.load('onehotlabel.npy')

all_imgs = np.concatenate([adv_imgs[:7000], data.x_train[:13000]])
all_labels = np.concatenate([adv_labels[:7000], data.y_train[:13000]])


num_iterations=2000

for i in range(num_iterations):
    idx = np.random.randint(low=0, high=len(all_imgs), size=batch_size)

    train_imgs=all_imgs[idx]
    train_labels= all_labels[idx]
    
    feed_dict_train = {x:train_imgs, y_true: train_labels}
    
    session.run(optimizer, feed_dict=feed_dict_train)
       

        # Print status every 100 iterations.
    if (i % 100 == 0) or (i == num_iterations - 1):
        # Calculate the accuracy on the training-set.
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
        print(msg.format(i, acc))
            

save_dir = 'checkpoints/adv/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

saver.save(sess=session, save_path=save_dir)

'''

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

save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

saver.save(sess=session, save_path=save_dir)
'''