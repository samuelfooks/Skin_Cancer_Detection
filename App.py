
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from streamlit_cropper import st_cropper
from PIL import Image, ImageOps
import numpy as np
 
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('data/modelling/my_saved_models/Xception2023-02-05_run3/model_weights/XceptionRandomOverSampleauto_dropout0.05_epochs200.98acc.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.title("""
         Mole Classification Application
         """
         )
st.subheader("Using Computer Vision and TensorFlow")

#function to upload the images and model
def upload_predict(upload_image, model):
    class_names=[0,1]
    # plt.imshow(load_img(upload_image), target_size=(150, 150)))
    # plt.show()
    size = (150,150)    
    image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
    #img = load_img(upload_image, target_size=(150, 150))
    img_array = []
    img = img_to_array(image)
    img = img.astype(np.float32) / 255
    img_array.append(img)

    img_array = np.stack(img_array, axis=0)

    predictions = model.predict(img_array)
    pred = np.argmax(predictions, axis=1)
    print(predictions, pred)
    prediction_prob = f"{(np.max(predictions))*100:.2f}%"
    if class_names[pred[0]] == 1:
        diagnosis = 'Cancerous'
    
    elif class_names[pred[0]] == 0:
        diagnosis = 'Non-Cancerous'
    
    return prediction_prob, diagnosis

#menu to choose upload or take a photo
choosen = st.radio(
    "Choose:",
    ('Upload Photo', 'Take Picture Now'))
#upload a photo
if choosen == 'Upload Photo':
    file = st.file_uploader("Upload the image to be classified", type=["jpg", "jpeg", "png"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
#enable taking a photo using the webcam
elif choosen == 'Take Picture Now':
    file = st.camera_input("Take a picture")

if file is None:
    st.text("Please provide an Image")
#open the image, display it, run the model 
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction_prob,diagnosis = upload_predict(image, model)

    st.write(diagnosis)
    st.write("Probability: " + prediction_prob)

    if diagnosis == 'Cancerous':
        st.write('Book a consultation with a doctor')
        st.button('Search my Area')
    else:
        st.write('Not a concern yet, keep wearing sunscreen and keep an eye on it!')

