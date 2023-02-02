
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2

from PIL import Image, ImageOps
import numpy as np
 
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('data/modelling/my_saved_models/RandomUnderSamplerXception_dropout0.15_epochs12_{accuracy:.2f}acc')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Image Classification
         """
         )
 
file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "jpeg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
 
def upload_predict(upload_image, model):
    class_names=[0,1]
    # plt.imshow(load_img(upload_image), target_size=(150, 150)))
    # plt.show()
    size = (150,150)    
    #image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
    img = load_img(upload_image, target_size=(150, 150))
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

    #print(f"\nPredicting: {(np.max(predictions))*100:.2f}% of {class_names[pred[0]]}")
    # if class_names[pred[0]] != 1:
    #     print("Kind of Cancer: Benign")
    #     print("Benign means it is not a dangerous cancer or not a cancer")
    # else:
    #     print("Kind of Cancer: Malignant")
    #     print("Malignant means dangerous and deadliest cancer")


    # size = (180,180)    
    # image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
    # image = np.asarray(image)
    # #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = cv2.resize(img, dsize=(150, 150),interpolation=cv2.INTER_CUBIC)
    
    # img_reshape = img_resize[np.newaxis,...]

    # prediction = model.predict(img_reshape)
    # print(prediction)
    #pred_class=decode_predictions(prediction,top=1)
    
    return prediction_prob, diagnosis
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction_prob,diagnosis = upload_predict(image, model)
    # image_class = str(predictions)
    # score=np.round(predictions) 
    # st.write("The image is classified as",image_class)
    # st.write("The similarity score is approximately",score)
    # print("The image is classified as ",image_class, "with a similarity score of",score)

    st.write(diagnosis)
    st.write(prediction_prob)
    print(diagnosis, "with a probability of ",prediction_prob)