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

data_set = pd.read_csv('train.csv', sep = ',', nrows = 100);
data_set.pickup_datetime = pd.DatetimeIndex(data_set.pickup_datetime).asi8 // 10**9
input = data_set.to_numpy()

train_out = input[:, 1:2]

train = np.array(input[:, 2:8])