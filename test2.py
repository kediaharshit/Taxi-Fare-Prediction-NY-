#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:34:20 2020

@author: hk3
"""

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


data_set = pd.read_csv('train.csv', sep = ',', nrows = 12000);
data_set.pickup_datetime = pd.DatetimeIndex(data_set.pickup_datetime).asi8 // 10**9
input = data_set.to_numpy()

x_train = input[0:10000, 2:8,]
y_train = input[0:10000, 1]

x_test = input[10000:12000, 2:8]
y_test = input[10000:12000, 1]

#x_train, x_test, y_train, y_test = train_test_split(input[:, 2:8], input[:, 1:2], test_size=0.2)

#Tensorflow placeholders - inputs to the TF graph
inputs =  tf.placeholder(tf.float32, [None, 6], name='Inputs')
targets =  tf.placeholder(tf.float32, [None, 1], name='Targets')

#Helper functions to define weights and biases

def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, 0.05, 0.05))

def init_biases(shape):
    return tf.Variable(tf.zeros(shape))

def fully_connected_layer(inputs, input_shape, output_shape, activation=tf.nn.relu):
    '''
    This function is used to create tensorflow fully connected layer.
    
    Inputs: inputs - input data to the layer
            input_shape - shape of the inputs features (number of nodes from the previous layer)
            output_shape - shape of the layer
            activatin - used as an activation function for the layer (non-liniarity)
    Output: layer - tensorflow fully connected layer
    
    '''
    #definine weights and biases
    weights = init_weights([input_shape, output_shape])
    biases = init_biases([output_shape])
    
    #x*W + b <- computation for the layer values
    layer = tf.matmul(inputs, weights) + biases
    
    #if activation argument is not None, we put layer values through an activation function
    if activation != None:
        layer = activation(layer)
        
    return layer


#defining the network
l1 = fully_connected_layer(inputs, 6, 16)
l2 = fully_connected_layer(l1, 16, 16)
l3 = fully_connected_layer(l2, 16, 1, activation=None)

predictions = l3
cost = loss2 = tf.reduce_mean(tf.squared_difference(targets, predictions))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

epochs = 10000
batch_size = 50
from tqdm import tqdm

#Starting session for the graph
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    #TRAINING PORTION OF THE SESSION
    for i in tqdm(range(epochs)):
        idx = np.random.choice(len(x_train), batch_size, replace=True)
        x_batch = x_train[idx, :]
        y_batch = y_train[idx]
        y_batch = np.reshape(y_batch, (len(y_batch), 1))
        
        batch_loss, opt = sess.run([cost, optimizer], feed_dict={inputs:x_batch, targets:y_batch})
        
        if i % 1000 == 0:
            print(batch_loss)
    
'''
    #TESTING PORTION OF THE SESSION
    preds = sess.run([predictions], feed_dict={inputs:x_test})
    true_preds = []
    for i in range(len(preds)):
        if preds[0][i] >= 0.5:
            true_preds.append(1)
        else:
            true_preds.append(0)
    
    true_correct = 0
    for i in range(len(preds)):
        if true_preds[i] == y_test[i]:
            true_correct += 1
    
    print("Accuracy: ", true_correct/len(true_preds))
'''