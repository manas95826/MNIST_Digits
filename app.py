import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Load the pre-trained MNIST digits model
model = keras.models.load_model('model.h5')

# Streamlit app title and description
st.title("MNIST Handwritten Digit Predictor")
st.write("Upload an image of a handwritten digit and I'll predict the number!")

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match MNIST image dimensions
    image = np.array(image)  # Convert to numpy array
    image = image.reshape(1, 28, 28, 1)  # Reshape to match model input shape
    image = image.astype('float32') / 255.0  # Normalize pixel values
    return image

# Upload image and make predictions
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and make prediction
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    
    st.write(f"Predicted Digit: {predicted_class}")
