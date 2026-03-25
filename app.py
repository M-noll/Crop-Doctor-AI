import os

# This automatically gets the names of your folders in alphabetical order
DATA_DIR = "plantvillage dataset/"
CLASS_NAMES = sorted(os.listdir(DATA_DIR))

# Print them once in your terminal just to double-check
print(CLASS_NAMES)
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('plant_doctor_model.h5')

# Define your classes (Make sure these match your folder names exactly!)
CLASS_NAMES = ['Tomato_Early_Blight', 'Tomato_Healthy', 'Potato_Late_Blight'] # Add all your classes here

st.title("🌱 AI Plant Doctor")
st.write("Upload a photo of a leaf to diagnose the disease.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    result = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"Diagnosis: **{result}**")
    st.info(f"Confidence: {confidence:.2f}%")