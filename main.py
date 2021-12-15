from cv2 import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications import InceptionResNetV2
import tensorflow_hub as hub

model = tf.keras.models.load_model(("transfer.h5"),custom_objects={'KerasLayer':hub.KerasLayer})

html_temp = """ 
    <div style ="background-color:#9B9BCB;padding:1px"> 
    <h1 style ="color:black;text-align:center;">Is it snowing or raining ?</h1> 
    </div> 
    """

st.markdown(html_temp, unsafe_allow_html=True)
st.image("Capture.JPG")
st.subheader('by Paul-Yuthi Lajus ')

uploaded_file = st.file_uploader("Choose an image file", type="jpg")

map_dict = {0: 'snowing',
            1: 'raining',
            }


if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Generate_pred = st.button("Generate Prediction")
    if Generate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("In this photo, it is {}".format(map_dict [prediction]))
