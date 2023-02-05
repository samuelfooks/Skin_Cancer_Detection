
#This file performs the preliminary Exploratory Data Analysis and provides options for alterations and refinement in the metadata
# To remove all warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from shutil import copy
from datetime import datetime

def process_data(csv, path, manual_downsample=False, sample_size=0):
    # Resizing images and moving function
    def resize_img(SOURCE, DEST, SIZE=299):
        files = []
        for filename in os.listdir(SOURCE):
            file = SOURCE + filename
            if os.path.getsize(file) > 0:
                files.append(filename)
            else:
                print(filename + " is zero length, so ignoring.")
        print(len(files))
        for filename in files:
            if '.jpg' in filename:
                img = cv2.imread(f"{SOURCE}{filename}")
                resize_img = cv2.resize(img, (SIZE,SIZE))
                cv2.imwrite(f"{SOURCE}/{filename}", resize_img)
                copy(f"{SOURCE}/{filename}",f"{DEST}/{filename}")

    try:
        os.mkdir('data/images/modelling_images')
        os.mkdir(f'{os.getcwd()}/data/images/resize_HAM10000') 
        
         ##Resizing the images 299 x 299
        resize_img(f'{os.getcwd()}/data/HAM10000_images_part_1/',f'{os.getcwd()}/data/resize_HAM10000/')
        resize_img(f'{os.getcwd()}/data/HAM10000_images_part_2/',f'{os.getcwd()}/data/resize_HAM10000/')
    except:
        pass

    skin_df = pd.read_csv(path + csv)
    resize_images_path = path + 'images/resize_HAM10000/'
    modelling_images_path = 'data/images/modelling_images/'
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # Creating New Columns for better readability
    skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 

    skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

    skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
    skin_df.isnull().sum()

    fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
    skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)

    #Make new column, risky  = 1, not risky = 0
    skin_df['Risk'] = skin_df['cell_type'].apply(lambda x : 'Not Risky' if ((x == 'Melanocytic nevi') | (x == 'Benign keratosis-like lesions') | (x == 'Dermatofibroma') | (x == 'Vascular lesions')) else 'Risky')
    skin_df['Risk'] = pd.Categorical(skin_df['Risk']).codes

    skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes


    fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
    fig.show()
    skin_df['Risk'].value_counts().plot(kind='bar', ax=ax1)

    skin_df.groupby(['Risk','cell_type']).size()

    #optional function to manually downsample a specific category of data from within a given feature in the dataset
    def downsample(skin_df, feature, category, samples):

        # Because too many Melanocytic Nevi, can downsample manually
        df_random = skin_df[skin_df[feature] == category].sample(n=samples, random_state=1)
        skin_df = skin_df.drop(skin_df[skin_df[feature]== category].index)
        skin_df = skin_df.append(df_random)

        #make new csv for downsampled dataframe
        skin_df.to_csv('data/' + datetime.today().strftime('%Y-%m-%d') + '_metadata_manual_downsample.csv')

        #copy the images of the new dataset into a different folder
        for f in os.listdir(modelling_images_path):
            os.remove(os.path.join(modelling_images_path, f))

        df_read = pd.read_csv('data/' + datetime.today().strftime('%Y-%m-%d') + '_metadata_manual_downsample.csv')
        for i in range(len(df_read)):
          copy(resize_images_path + f"{df_read['image_id'].values[i]}.jpg", modelling_images_path + f"{df_read['image_id'].values[i]}.jpg")
        return

    if manual_downsample==True:
        downsample(skin_df, feature = 'cell_type', category='Melanocytic nevi', samples = sample_size)
    
    #else write the new_metadata with Risky column to a csv to be further processed
    elif manual_downsample==False:
        skin_df.to_csv('data/' + datetime.today().strftime('%Y-%m-%d') + '_new_metadata.csv')
        for i in range(len(skin_df)):
            copy(resize_images_path + f"{skin_df['image_id'].values[i]}.jpg", modelling_images_path + f"{skin_df['image_id'].values[i]}.jpg")
    return 

process_data('HAM10000_metadata.csv', 'data/', manual_downsample=True, sample_size=2410)   
print(len(os.listdir('data/images/modelling_images'))) 