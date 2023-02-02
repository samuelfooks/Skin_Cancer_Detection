# To remove all warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import zipfile
import pandas as pd
import requests
import random

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def process_split(csv, oversample=False, undersample=False):

    def define_dataset(csv, unique=False):
            df = pd.read_csv(csv)
            img = df['image_id'] 
            target = df['Risk'].values 
            if unique: # Counting unique values
                df['cell_type'].value_counts() # Returning amout of unique value
            return img, target
    
    def split_data(x, y, test_size=0.15, validation_size=0.15, oversample=False, undersample=False):
        # Balancing the datasets, if the datasets not balanced so this function will balanced them
        def balancing_dataset(x_train, y_train, oversample=False, undersample=False):
            if oversample:
                oversample= RandomOverSampler(sampling_strategy=0.5)
                x_train,y_train = oversample.fit_resample(x_train.values.reshape(-1,1),y_train)
            if undersample:
                undersample = RandomUnderSampler(sampling_strategy=0.9)
                x_train, y_train  = undersample.fit_resample(x_train.values.reshape(-1,1),y_train)
            return x_train, y_train
        
        #split into train, test, validate
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)

        #apply over or undersampling on x_train and y_train
        if oversample:
            x_train, y_train = balancing_dataset(x_train, y_train, oversample=True, undersample=False)
        if undersample:
            x_train, y_train = balancing_dataset(x_train, y_train, oversample=False, undersample=True)
        # Print the results of splitting
        print("Train: ", x_train.shape[0]), print("Val: ", x_val.shape[0]), print("Test: ", x_test.shape[0]), 
        return x_train, x_val, x_test, y_train, y_val, y_test

    img, target = define_dataset(csv)
    # Splitting into train, valid and test with upsampling or downsampling all of the data
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(img, target, oversample=oversample, undersample=undersample, test_size=0.2, validation_size=0.15)

    y_train_series = pd.Series(y_train)
    print(y_train_series.value_counts())


    #convert the images to vectors, default 150 * 150.  Then rescale by 255
    def to_tensor(image_paths, oversample=False, undersample=False, size=150):
        imgs = []
        for i in tqdm(image_paths):
            if oversample or undersample: # If datasets is balanced
                img = load_img(f"data/images/resize_HAM10000/{i[0]}.jpg", target_size=(size, size)) # Load the Image then resized them
            else:
                img = load_img(f"data/images/resize_HAM10000/{i}.jpg", target_size=(size, size)) # Load the Image then resized them

            img = img_to_array(img) # Convert the Image to arrays
            img = img.astype(np.float32) / 255 # Rescale the Images
            imgs.append(img) # Load all of Images to datasets
        imgs = np.stack(imgs, axis=0) # Stack the image into numpy arrays    
        return imgs
    
    #Here must specify if either undersample or oversample has been done, in order to process the array output
    if oversample or undersample:
        x_train = to_tensor(x_train, oversample=True, size=150)
    else:
        x_train = to_tensor(x_train, size=150)   
    x_val = to_tensor(x_val, size=150)
    x_test = to_tensor(x_test, size=150)

    return x_train, x_val, x_test, y_train, y_val, y_test

#data_process_split('/data/new_metadata.csv', undersample=True)