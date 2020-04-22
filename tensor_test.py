#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:57:53 2020

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

x_train, x_test, y_train, y_test = train_test_split(input[:, 2:8], input[:, 1:2], test_size=0.2)



learning_rate = 0.01
epochs = 10
batch_size = 10

x = tf.placeholder(dtype = tf.float32, shape = [None, 6], name = 'x')
y = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = 'y')

# first hidden later
h1_size = 10
w1 = tf.Variable(tf.random_normal([h1_size, 6, stddev=0.09), name='w1')
b1 = tf.Variable(tf.constant(0.1, shape=(h1_size, 1)), name='b1')
y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)))

# Second hidden layer
h2_size = 10
w2 = tf.Variable(tf.random_normal([h2_size, h1_size], stddev=0.09), name='w2')
b2 = tf.Variable(tf.constant(0.1, shape=(h2_size, 1)), name='b2')
y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)))

# Third hidden layer
h3_size = 10
w3 = tf.Variable(tf.random_normal([h3_size, h2_size], stddev=0.09), name='w3')
b3 = tf.Variable(tf.constant(0.1, shape=(h3_size, 1)), name='b3')
y3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w3, y2), b3)))

# Output layer
wo = tf.Variable(tf.random_normal([1, h3_size], stddev=0.09), name='wo')
bo = tf.Variable(tf.random_normal([1, 1]), name='bo')
yo = tf.transpose(tf.add(tf.matmul(wo, y2), bo))

lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
loss = tf.reduce_mean(tf.square(y_pred - y)) # Square Loss Error Function
optimizer = tf.train.AdamOptimizer(0.1).ninimize(loss) # AdamOptimiser


