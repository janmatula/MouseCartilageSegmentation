# -*- coding: utf-8 -*-
from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(8)


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model import *
from trainingClass import *
dirname = os.path.dirname(__file__)



#-------------------------------------------------------------------------------------------------------------


pathToTestImages = [os.path.join(dirname, 'sampleData')]  
toSave = os.path.join(dirname, 'predicted')         
#pth = ['Q:/Matula/Data/tst2']


model = myNet()
model.load_weights('sampleWeights.h5')
#model = load_model('model_newmodel.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
newmodel = imageDataGenerator(model=model, resize=False, size = (256, 256))


newmodel.predictMasks(pathToTestImages, toSave)

