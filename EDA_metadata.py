
#This file performs the preliminary Exploratory Data Analysis and provides options for alterations and refinement in the metadata
# To remove all warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from shutil import copy


def process_data(csv, path):
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
        os.mkdir(f'{os.getcwd()}/data/images/resize_HAM10000') 

         ##Resizing the images 299 x 299
        resize_img(f'{os.getcwd()}/data/HAM10000_images_part_1/',f'{os.getcwd()}/data/resize_HAM10000/')
        resize_img(f'{os.getcwd()}/data/HAM10000_images_part_2/',f'{os.getcwd()}/data/resize_HAM10000/')
    except:
        pass

    skin_df = pd.read_csv(path + 'HAM10000_metadata.csv')
    image_path = path + '/images/resize_HAM10000'

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
    def manual_downsample(skin_df, feature, category, samples):

        # Because too many Melanocytic Nevi, can downsample manually
        df_random = skin_df[skin_df[feature] == category].sample(n=samples, random_state=1)
        skin_df = skin_df.drop(skin_df[skin_df[feature]== category].index)
        skin_df = skin_df.append(df_random)

        #make new csv for downsampled dataframe
        skin_df.to_csv('data/metadata_downsample.csv')

        #copy the images of the new dataset into a different folder
        try:
            os.mkdir(f'{os.getcwd()}/data/images/images_downsample')
        except:
            pass
        df_read = pd.read_csv('data/metadata_downsample.csv')
        for i in range(len(df_read)):
          copy(f"data/images/resize_HAM10000/{df_read['image_id'].values[i]}.jpg", f"data/images/images_downsample/{df_read['image_id'].values[i]}.jpg")
        
        return

    #if manually downsampling dataframe uncomment here, choose sample number to take from dominant result, (in this case 2410 from cell_type M. nevi)
    #manual_downsample(skin_df, feature = 'cell_type', category='Melanocytic nevi', 2410)
    #skin_df.to_csv('data/metadata_downsample.csv')

    #else write the new_metadata with Risky column to a csv to be further processed
    skin_df.to_csv('data/new_metadata.csv')

    return 
    