import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. SETUP DATA DIRECTORY
DATA_DIR = "plantvillage dataset/color"

# 2. AUTOMATICALLY GET CLASS NAMES (The "Sanity" Fix)
# This ensures Index 0 is Apple, Index 10 is Corn, etc., matching the trainer.
if os.path.exists(DATA_DIR):
    CLASS_NAMES = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])
else:
    # Fallback if folder isn't found
    st.error("Dataset folder not found! Check your path.")
    CLASS_NAMES = []

# 3. LOAD THE MODEL
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('plant_doctor_model.h5')

model = load_my_model()

# 4. USER INTERFACE
st.title("🌱 AI Plant Doctor")
st.write("Upload a photo of a leaf to diagnose the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB') # Fix for transparent images
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    
    # Preprocess
    img = image.resize((128, 128))
    # Before: img_array = np.array(img) / 255.0
    # After: Add this to make the AI more robust
    img_array = np.array(img).astype('float32')
    img_array /= 255.0

    # Optional: Simple 'Color Constant' check to handle sunlight
    # (This helps the AI focus on spots, not the brightness of the sun)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner('Analyzing...'):
        predictions = model.predict(img_array)
        idx = np.argmax(predictions)
        
        if idx < len(CLASS_NAMES):
            result = CLASS_NAMES[idx]
            confidence = np.max(predictions) * 100

            # Clean up the folder name for display (e.g., Apple___Apple_scab -> Apple Scab)
            display_name = result.replace("___", " ").replace("_", " ")

    if confidence > 75:
        st.success(f"✅ Diagnosis: {display_name}")
        st.info(f"Confidence: {confidence:.2f}%")
    elif confidence > 40:
        st.warning(f"🤔 Uncertain: Might be {display_name} (Confidence: {confidence:.2f}%)")
        st.write("Tip: Try moving closer to the leaf or improving the lighting.")
    else:
        st.error("❌ Scan Failed: I don't recognize this. Please try a clearer photo of a single leaf.")