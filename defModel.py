#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:41:25 2020

@author: Ines
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import LeakyReLU
from keras.layers.core import Flatten
from keras.layers.core import Dense
import tensorflow as tf

def Model(INPUT_SHAPE,metric_iou):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), padding = 'same', input_shape=(INPUT_SHAPE)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(16, kernel_size=(3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, kernel_size=(3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(32, kernel_size=(3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=(3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
     
    model.add(Conv2D(256, kernel_size=(3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=(3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten()) 
    
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(6)) 
    #bbox
    #model_outputsB = model.add(Dense(4))
    #size
    #model_outputsS = model.add(Dense(2))
    
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),loss='mean_squared_error',metrics=[metric_iou])
    
    
    return model