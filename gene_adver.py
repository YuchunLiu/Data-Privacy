#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:04:33 2018

@author: yuchunliu
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv

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

net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=32, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=64, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.contrib.layers.flatten(net)

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
    return predict_cls(images = attack_img,
                       labels = attack_label,
                       cls_true = att_label_num)
    
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
    


def top_five(precon, orig_label):
    count = 0
    for i in range(len(precon)):
        five_acc=np.argsort(precon[i])[-5:]
        
        if orig_label[i] in five_acc:
            count +=1
        #print("The result is")
        #print(orig_label[i],five_acc)
    acc = count/len(orig_label)
    return acc

def top_one(precls, orig_label):
    count = 0
    for i in range(len(orig_label)):
        if precls[i]==orig_label[i]:
            count+=1
    acc = count/ len(orig_label)
    return acc



def target_iter(image, label, stsize, target_class):
    
    one_hot_label = np.zeros((len(image), 10))
    one_hot_label[np.arange(len(image)), target_class]=1.0
    
   
    
    grad = session.run(gradient, feed_dict = 
                                     {x: image, y_true: one_hot_label})
   
    adver_image = image - stsize * np.sign(grad)
    pred_cls,pred_con = session.run([y_pred_cls,y_pred], feed_dict = 
                                     {x: adver_image, y_true: label})
 
    return adver_image, pred_cls


def basic_iter(image, label, stsize):
    
    
    grad = session.run(gradient, feed_dict = 
                                     {x: image, y_true: label})
   
    adver_image = image + stsize * np.sign(grad)
    pred_cls = session.run(y_pred_cls, feed_dict = 
                                     {x: adver_image, y_true: label})
 
    return adver_image, pred_cls


save_dir = 'checkpoints/'
saver = tf.train.Saver()
session = tf.Session()
saver.restore(sess=session, save_path=save_dir)



attack_idx = np.random.choice(len(data.y_test), 10000)
attack_img = data.x_test[attack_idx]
attack_label = data.y_test[attack_idx]
att_label_num = data.y_test_cls[attack_idx]

print_test_accuracy()

target_adv, pred_cls= basic_iter(attack_img, attack_label, 0.2)


'''
for i in range(len(target_adv)):
    adv_path= './data/ADV/'
    norm_path = './data/NORM/'
    img_path = str(i)+'.png'
    cv.imwrite(adv_path+img_path, np.reshape(255*target_adv[i], (28,28)))
    cv.imwrite(norm_path+img_path, np.reshape(255*attack_img[i], (28,28)))
'''

np.save('adv.npy', target_adv)
np.save('onehotlabel.npy',attack_label)
    