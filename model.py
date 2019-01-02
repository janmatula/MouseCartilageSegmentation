# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 07:27:18 2018

@author: jan.matula
"""
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

alpha = 0.3
beta = 0.7
def Tversky(y_true, y_pred):
    smooth=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    G_P = alpha*K.sum( (1-y_true_f) * y_pred_f  ) # G not P
    P_G = beta*K.sum(  y_true_f * (1-y_pred_f) )  # P not G
    return (intersection + smooth )/(intersection + smooth +G_P +P_G)
def Tversky_loss(y_true, y_pred):
    return -Tversky(y_true, y_pred)

def myNet(input_size = (256,256,1)):
    #pretrained part
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv5)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv5)
    
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    #randomly initialized part
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(pool5)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv6)
    
    up7 = Conv2DTranspose(256, (2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv6)
    merge7 = concatenate([conv5,up7],axis=3)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(merge7)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv7)
    
    up8 = Conv2DTranspose(256,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv7)
    merge8 =concatenate([conv4,up8],axis=3)
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(merge8)
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv8)
    
    up9 = Conv2DTranspose(128,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv8)
    merge9 =concatenate([conv3,up9],axis=3)
    conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(merge9)
    conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv9)
    
    up10 = Conv2DTranspose(64,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv9)
    merge10 =concatenate([conv2,up10],axis=3)
    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(merge10)
    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv10)
    
    up11 = Conv2DTranspose(32,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv10)
    merge11 =concatenate([conv1,up11],axis=3)
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(merge11)
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv11)
    
    conv12 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal')(conv11)
    conv12 = Conv2D(1, 1, activation = 'sigmoid')(conv12)

    model = Model(input = inputs, output = conv12)
    
    model.compile(optimizer = Adam(lr = 1e-4, amsgrad = True), loss = Tversky_loss, metrics = [Tversky])
    
    return model