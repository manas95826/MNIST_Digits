import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained MNIST digits model
model = keras.models.load_model('model.h5')
import os

# Set environment variable to enable OpenCV headless mode
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

st.title("Handwritten Digit Classifier")
# Create a canvas for drawing
canvas = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
)
classify_button = st.button("Classify")

if classify_button:
    # Get the drawing from the canvas
    img_data = canvas.image_data.astype(np.uint8)
    img = np.array(img_data)
    greyscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    input_image_resized = cv2.resize(greyscale, (28,28))
    input_reshape = np.reshape(input_image_resized, [1,28,28])
    #predict_image = preprocessing(img)

    
    # Perform prediction
    prediction = predict_digit(model, input_reshape)
    print(prediction)

    st.write("Prediction:", prediction)
