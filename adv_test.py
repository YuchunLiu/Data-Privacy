#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:50:00 2018

@author: yuchunliu
"""
import tensorflow as tf
import numpy as np
import cv2 as cv
import os

from mnist import MNIST
data = MNIST(data_dir="data/MNIST/")

img_size = data.img_size
img_size_flat = data.img_size_flat

img_shape = data.img_shape
num_classes = data.num_classes
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

batch_size = 256

def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)
       
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = data.x_test[:3000],
                       labels = data.y_test[:3000],
                       cls_true = data.y_test_cls[:3000])
    
def cls_accuracy(correct):
    correct_sum = correct.sum()
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

def print_test_accuracy():  
    correct, cls_pred = predict_cls_test()
    acc, num_correct = cls_accuracy(correct)
    num_images = len(correct)
    
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

def read_img():
    '''
    dir_path = os.listdir('./data/ADV')
    adv_imgs = []
    for i in range(len(dir_path)):
        img = cv.imread('./data/ADV/'+dir_path[i],0)
        img_flat =img.reshape(-1,784)[0]/255.0
        adv_imgs.append(img_flat)
    
    adv_imgs = np.asarray(adv_imgs)
    '''
    adv_imgs = np.load('adv.npy')
    adv_labels = np.load('onehotlabel.npy')

    #all_imgs = np.concatenate([adv_imgs, data.x_train])
    #all_labels = np.concatenate([adv_labels, data.y_train])
    all_imgs = adv_imgs[7000:]
    all_labels = adv_labels[7000:]
    
    all_label_cls = []
    
    for i in all_labels:
        all_label_cls.append(np.argmax(i))
    return all_imgs, all_labels, all_label_cls

all_imgs, all_labels, all_label_cls= read_img()



save_dir = 'checkpoints/adv/'
saver = tf.train.Saver()
session = tf.Session()
saver.restore(sess=session, save_path=save_dir)



print_test_accuracy()
