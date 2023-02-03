
#This file will build, train and evaluate a neural network and apply transfer learning using the dataset provided from
#data_process_split.  The option is there to build you own "base_model" in layers, dropouts, etc or use the most
#successful models from my testing, 'Xception', 'MobileNetV2', and 'VGG16'. Part one is the training and saving of the model
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


#create dataset using process_split function, chose undersample or oversample if desired, sample_strategy is 0.5 by default
# test and validation data proportions are 0.2 by default

oversample=True
undersample=False
sample_strategy = 0.3

x_train, x_val, x_test, y_train, y_val, y_test = process_split('data/new_metadata.csv', oversample=oversample, sample_strategy=sample_strategy)


# Prediction on the test portion
def predictions(model, x_test, y_test, accuracy=True, axis=1):
  
  predictions = model.predict(x_test) # Predict the test datasets
  pred = np.argmax(predictions, axis=axis) # Showing the greatest value from predictions (range: 0-1, biggest is 1)
  
  # Printing the accuracy, with comparing predictions with labels of test datasets portion
  if accuracy: 
    print("\nAccuracy: {0:.2f}%".format(accuracy_score(y_test, pred)*100))
  
  return pred, x_test, y_test

#function to train the model, choice of many pre trained models to apply transfer learning on
#if you dont want to use transfer learning, set include_top=True
#based on github.com/faniabdullah/bangkit-final-project/tree/master/Machine%20Learning%20Development/testing

#make a function to train the model. Many parameters here to be tested based on user
#this function provides also the image data generator function to help the model train on images that are turned, flipped, zoomed out, etc
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
#setting include_top=True will freeze the last layer of the base_model

epochs = 20
dropout = 0.15
batch_size = 32
learning_rate= 1e-4
fine_tuning = True
include_top= False
base_model= 'Xception'

callbacks = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

#create a directory and store the model weights based on parameters chosen, to keep track of trained models
#in future versions a link with ML flow will be used here instead
if oversample:
  sampling = 'RandomOverSample'
elif undersample:
  sampling = 'RandomUnderSample'

try:
  os.mkdir('data/modelling/model_weights/' +  base_model + sampling + str(sample_strategy) + '_dropout' +  str(dropout) + '_epochs' + str(epochs) + "_03022023")
except:
  pass

model_weights_run= 'data/modelling/model_weights/'

checkpoints = ModelCheckpoint(model_weights_run + base_model + sampling + str(sample_strategy) + '_dropout' +  str(dropout) + '_epochs' + str(epochs) + "{accuracy:.2f}acc" + "_03022023_run2.h5", verbose=1)

#train the model and display stats per epoch, save the weights, and monitor the callbacks to stop early if necessary
history, model = trainable_model(x_train, y_train, x_val, y_val, x_test, y_test,
                                 fine_tuning=fine_tuning, epochs=epochs, base_model=base_model, 
                                 dropout=dropout, regularizer=0.1, batch_size=batch_size,
                                 callbacks=callbacks, summary=True, checkpoint=checkpoints)

#save the model for use in the App.py
try:
  saved_model_dir = os.mkdir('data/modelling/my_saved_models/')
except:
  pass
model.save(saved_model_dir + base_model + sampling + str(sample_strategy)+ '_dropout' +  str(dropout) + '_epochs' + str(epochs) + " {accuracy:.2f}acc" + '_03022023_run2')

#This part will evaluate the performance of the models predictions vs the test data

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

def plot_confusionmat(model_plot_dir, model_plot, confusion_mat, class_names, cmap='GnBu'):
  
  fig, ax = plt.subplots(figsize=(10,10))
  
  sns.heatmap(confusion_mat, annot=True, fmt='.2f',
            xticklabels=[f"{c}" for c in class_names], 
            yticklabels=[f"{c}" for c in class_names],
            cmap=cmap)
  
  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  plt.savefig(model_plot_dir + model_plot)
  return plt.show()

class_names = [0,1]
try:
  os.mkdir('data/saved_models/plots')
except:
  pass 
model_plot_dir = 'data/saved_models/plots/'
model_plot = base_model + sampling + str(sample_strategy)+ '_dropout' +  str(dropout) + '_epochs' + str(epochs) + " {accuracy:.2f}acc" + '_03022023' + 'confusion_matrix.png'
plot_confusionmat(confusion_mat, class_names)