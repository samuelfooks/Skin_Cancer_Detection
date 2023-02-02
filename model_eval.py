
# To remove all warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import zipfile
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.applications import MobileNet, MobileNetV2, VGG16, EfficientNetB0, InceptionV3, \
                                           VGG19, Xception, DenseNet121, DenseNet201, ResNet152V2, EfficientNetB5
from tensorflow.keras.layers import Input, AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model
from tensorflow import lite, cast, float32
from tensorflow import saved_model

from data_process_split import process_split 

x_train, x_val, x_test, y_train, y_val, y_test = process_split('data/new_metadata.csv', undersample=True)


# dropout= 0.15
# dataset = (x_train, y_train, x_val, y_val)
# pool_size = 2
# regularizer = 0.01
# BATCH_SIZE = 32
# INPUT_SHAPE = dataset[0][0].shape
# weights = 'imagenet'
# include_top=False
# base_model = Xception(weights=weights, 
# include_top=include_top, 
# input_tensor=Input(shape=INPUT_SHAPE))

# base_model = Xception(weights=weights, 
#                 include_top=include_top, 
#                 input_tensor=Input(shape=INPUT_SHAPE))


# epochs = 12
# dropout = 0.15
# batch_size = 32
# learning_rate= 1e-4
# fine_tuning = True


# model = Sequential([base_model, 
#                       AveragePooling2D(pool_size=(pool_size ,pool_size)),      
#                       Flatten(), 
#                       Dense(64, activation='relu'),
#                       Dropout(dropout), 
#                       Dense(16, activation='relu'),
#                       Dense(2, activation='softmax', kernel_regularizer=l2(regularizer)),
#   ])
  
# model.compile(optimizer=Adam(learning_rate=1e-4), 
#                 loss=SparseCategoricalCrossentropy(), 
#                 metrics=['accuracy'])
# #model.load_weights('data/modelling/model_weights/HAM10000_Xception_dropout015_0.98acc.h5')

# Scoring saved models/checkpoint models

#model = load_model('data/modelling/my_saved_models/RandomOverSamplerXception_dropout0.15_epochs12_{accuracy:.2f}acc')
model = load_model('data/modelling/my_saved_models/RandomUnderSamplerXception_dropout0.15_epochs12_{accuracy:.2f}acc')

def scoring(model, x_test, y_test, verbose=10, returning='confusion_matrix'):
  
  score = model.evaluate(x_test, y_test, verbose=verbose)
  predicting = model.predict(x_test)
  pred = np.argmax(predicting, axis=1)
  conf = confusion_matrix(y_test, pred)  
  
  if returning in ['score', 'scoring']:
    return score
  
  if returning in ['predicting', 'pred', 'predict']:
    return pred

  if returning in ['conf', 'confusion_matrix', 'confussion', 
                   'confusion', 'conf_mat', 'confusion_mat']:
    return conf


score = scoring(model, x_test, y_test, returning='score')
confusion_mat = scoring(model, x_test, y_test, returning='confusion_mat')

def plot_confusionmat(confusion_mat, class_names, cmap='GnBu'):
  
  fig, ax = plt.subplots(figsize=(10,10))
  
  sns.heatmap(confusion_mat, annot=True, fmt='.2f',
            xticklabels=[f"{c}" for c in class_names], 
            yticklabels=[f"{c}" for c in class_names],
            cmap=cmap)
  
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  return plt.show()

class_names = [0,1]

plot_confusionmat(confusion_mat, class_names)