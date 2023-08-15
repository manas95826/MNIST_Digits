import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained MNIST digits model
model = keras.models.load_model('model.h5')

# Streamlit app title and description
st.title("MNIST Handwritten Digit Predictor")
st.write("Upload an image of a handwritten digit or draw a digit on the canvas, and I'll predict the number!")

# Function to preprocess the uploaded image or canvas drawing
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to match MNIST image dimensions
    image = np.array(image)  # Convert to numpy array
    image = image.reshape(1, 28, 28, 1)  # Reshape to match model input shape
    image = image.astype('float32') / 255.0  # Normalize pixel values
    return image

# Streamlit drawing canvas
st.write("Draw a digit on the canvas:")
canvas = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
)

if st.button("Predict"):
    # Convert the canvas drawing to an OpenCV image
    drawn_image = np.array(canvas.image_data)
    
    # Preprocess and make prediction
    preprocessed_image = preprocess_image(drawn_image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    
    st.write(f"Predicted Digit: {predicted_class}")
