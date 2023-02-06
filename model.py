
#This file will build, train and evaluate a neural network and apply transfer learning using the dataset provided from
#data_process_split.  The option is there to build you own "base_model" in layers, dropouts, etc or use the most
#successful models from my testing, 'Xception', 'MobileNetV2', and 'VGG16'. Part one is the training and saving of the model
import pandas as pd
import csv
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, VGG16, Xception
from tensorflow.keras.layers import Input, AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow import lite, cast, float32
from tensorflow import saved_model

from data_process_split import process_split


# Prediction function to be used on the test portion
def predictions(model, x_test, y_test, accuracy=True, axis=1):
  
  predictions = model.predict(x_test) # Predict the test datasets
  pred = np.argmax(predictions, axis=axis) # Showing the greatest value from predictions (range: 0-1, biggest is 1)
  
  # Printing the accuracy, with comparing predictions with labels of test datasets portion
  if accuracy: 
    print("\nAccuracy: {0:.2f}%".format(accuracy_score(y_test, pred)*100))
  
  return pred, x_test, y_test

#function to train the model, choice of many pre trained base_models to apply transfer learning
#fine_tuning is set to False, but if turned to True will train the chosen base_model
#based on github.com/faniabdullah/bangkit-final-project/tree/master/Machine%20Learning%20Development/testing

#make a function to train the model. Many parameters here to be tested based on user
#this function also provides the image data generator function to help the model train on images that are turned, 
#flipped, zoomed out, etc
def trainable_model(x_train, y_train, x_val, y_val, x_test, y_test, batch_size=64, 
                    fine_tuning=False, dropout=0.25, base_model='MobileNet', 
                    regularizer=0.01, learning_rate=1e-4, epochs=15, verbose=1, 
                    metrics='accuracy', pool_size=2, rotation_range=30, 
                    zoom_range=0.15, width_shift_range=0.2, shear_range=0.15, 
                    horizontal_flip=True, fill_mode="nearest", height_shift_range=0.2,
                    weights="imagenet", include_top=False, summary=False, 
                    valid_generator=False, callbacks=None, generator=True,
                    checkpoint=None):
  dataset = (x_train, y_train, x_val, y_val)

  BATCH_SIZE = batch_size
  INPUT_SHAPE = dataset[0][0].shape
  trainX = dataset[0]
  trainY = dataset[1]
  valX = dataset[2]
  valY = dataset[3]

  generators = ImageDataGenerator()
  train_dataset = generators.flow(trainX, trainY, batch_size=BATCH_SIZE)
  valid_dataset = generators.flow(valX, valY, batch_size=BATCH_SIZE)

  if generator:
    train_gen = ImageDataGenerator(rotation_range=rotation_range,
                                zoom_range=zoom_range,
                                width_shift_range=width_shift_range,
                                height_shift_range=height_shift_range,
                                shear_range=shear_range,
                                horizontal_flip=horizontal_flip,
                                fill_mode=fill_mode)

    train_dataset = train_gen.flow(trainX, trainY, batch_size=BATCH_SIZE)
    
    if valid_generator:
      valid_dataset = train_gen.flow(valX, valY, batch_size=BATCH_SIZE)

  if base_model is 'Xception':
    base_model = Xception(weights=weights, 
                             include_top=include_top, 
                             input_tensor=Input(shape=INPUT_SHAPE)) # Xception: 95%, Epochs: 9 

  elif base_model is 'MobileNetV2':
    base_model = MobileNetV2(weights=weights, 
                             include_top=include_top, 
                             input_tensor=Input(shape=INPUT_SHAPE)) # MobileNetV2: 75%, Epochs: 8

  elif base_model is 'VGG16':
    base_model = VGG16(weights=weights, 
                       include_top=include_top, 
                       input_tensor=Input(shape=INPUT_SHAPE)) # VGG16: 90%, Epochs: 7
  base_model.trainable=False
  if fine_tuning:
    base_model.trainable=True
  #model architecture, can use pretrained base_model or not  
  model = Sequential([base_model, 
                      AveragePooling2D(pool_size=(pool_size ,pool_size)),      
                      Flatten(), 
                      Dense(64, activation='relu'),
                      Dropout(dropout), 
                      Dense(16, activation='relu'),
                      Dense(2, activation='softmax', kernel_regularizer=l2(regularizer)),
  ])

  model.compile(optimizer=Adam(lr=learning_rate), 
                loss=SparseCategoricalCrossentropy(), 
                metrics=[metrics])
  if summary:
    model.summary()
  history = model.fit(train_dataset, 
                      epochs=epochs, 
                      validation_data=valid_dataset, 
                      verbose=verbose,
                      callbacks=[callbacks, checkpoint])
  _ = predictions(model, x_test, y_test)
  return history, model

#chose number of epochs, batch and learning rate, fine_tuning= True allows the base model to train
#setting include_top=True will unfreeze the last layer of the base_model
epochs = 20
dropout = 0.05
batch_size = 32
learning_rate= 1e-4
fine_tuning = True
include_top= False
base_model= 'Xception'

#choose a point for earlystopping if using many epochs
callbacks = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

#make directories for all training data, model weights, and model

os.mkdir('data/modelling/my_saved_models/' + base_model + datetime.today().strftime('%Y-%m-%d') + '_run3/')
saved_model_dir = 'data/modelling/my_saved_models/' + base_model + datetime.today().strftime('%Y-%m-%d') + '_run3/'
os.mkdir(saved_model_dir + 'model_weights/')
model_weights_dir = saved_model_dir + '/model_weights/'

#choose sampling type and proportion for the splitting function
oversample=True
undersample=False
sample_strategy = 'auto'

# test and validation data proportions are 0.2 by default
x_train, x_val, x_test, y_train, y_val, y_test = process_split('data/2023-02-05_metadata_manual_downsample.csv', saved_model_dir, oversample=oversample, sample_strategy=sample_strategy)

if oversample:
  sampling = 'RandomOverSample'
elif undersample:
  sampling = 'RandomUnderSample'

#set checkpoints to save the model at each epoch
checkpoints = ModelCheckpoint(model_weights_dir + base_model + sampling + str(sample_strategy) + '_dropout' +  str(dropout) + '_epochs' + str(epochs) + "{accuracy:.2f}acc.h5", verbose=1)

#train the model and display stats per epoch, save the weights, and monitor the callbacks to stop early if necessary
history, model = trainable_model(x_train, y_train, x_val, y_val, x_test, y_test,
                                 fine_tuning=fine_tuning, epochs=epochs, base_model=base_model, 
                                 dropout=dropout, regularizer=0.1, batch_size=batch_size,
                                 callbacks=callbacks, summary=True, checkpoint=checkpoints)

#save the model for use in the App.py
model.save(saved_model_dir + base_model + sampling + str(sample_strategy)+ '_dropout' +  str(dropout) + '_epochs' + str(epochs) + " {accuracy:.2f}acc" + datetime.today().strftime('%Y-%m-%d') + 'run3')

#score the test data based on the trained model(or another saved model)
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

#run the scoring function and make a confusion matrix
score = scoring(model, x_test, y_test, returning='score')
confusion_mat = scoring(model, x_test, y_test, returning='confusion_mat')

#plot the results of the confusion matrix, save them to the model directory
def plot_confusionmat(confusion_mat, class_names, cmap='GnBu'):
  fig, ax = plt.subplots(figsize=(10,10))
  
  sns.heatmap(confusion_mat, annot=True, fmt='.2f',
            xticklabels=[f"{c}" for c in class_names], 
            yticklabels=[f"{c}" for c in class_names],
            cmap=cmap)
  
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.savefig(saved_model_dir + 'confusion_matrix.png')
  return plt.show()

class_names = [0,1]
plot_confusionmat(confusion_mat, class_names)