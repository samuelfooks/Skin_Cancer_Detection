%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools
import glob
import csv


import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

base_skin_dir = 'data/'

image_dir = 'data/HAM_images'
# This dictionary is useful for displaying more human-friendly labels later on

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
def metadata_preprocessing():
    
    
    skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

    fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
    skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)

    # Creating New Columns for better readability

    skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
    skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 

    skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
    
    skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
    print(skin_df.isnull().sum())

    fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
    skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)

    return skin_df