
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2

from PIL import Image, ImageOps
import numpy as np
 

def load_my_model():
  model=tf.keras.models.load_model('data/modelling/my_saved_models/multimodel_binarytarget_31012023')
  return model
# with st.spinner('Model is being loaded..'):
#   model=load_model()


def upload_predict(my_images_path, upload_image, model):
    class_names=[0,1]
    # plt.imshow(load_img(upload_image), target_size=(150, 150)))
    # plt.show()
    size = (150,150)    
    #image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
    img = load_img(my_images_path+ upload_image, target_size=(150, 150))
    img_array = []
    img = img_to_array(img)
    img = img.astype(np.float32) / 255
    img_array.append(img)

    img_array = np.stack(img_array, axis=0)

    predictions = model.predict(img_array)
    pred = np.argmax(predictions, axis=1)

    prediction_prob = f"{(np.max(predictions))*100:.2f}%"
    if class_names[pred[0]] == 1:
        diagnosis = 'Professional Examination Recommended'
    
    elif class_names[pred[0]] == 0:
        diagnosis = 'Unconcerning'

    print(upload_image, diagnosis, prediction_prob)
    return prediction_prob, diagnosis

my_model = load_my_model()
my_images_path='data/internet_images/'
for file in os.listdir(my_images_path):
    upload_predict(my_images_path, file, my_model)

