
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
  model=tf.keras.models.load_model('SAM_multimodel_binarytarget_31012023')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.title("""
         Mole Classification Application
         """
         )
st.subheader("Using Computer Vision and TensorFlow")


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

choosen = st.radio(
    "Choose:",
    ('Upload Photo', 'Take Picture Now'))

if choosen == 'Upload Photo':
    file = st.file_uploader("Upload the image to be classified", type=["jpg", "jpeg", "png"])
    st.set_option('deprecation.showfileUploaderEncoding', False)

elif choosen == 'Take Picture Now':
    file = st.camera_input("Take a picture")

if file is None:
    st.text("Please provide an Image")
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

