import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model.h5")  # Ensure "model.h5" is in the same directory

# Function to make predictions
def predict_forgery(img):
    img = img.resize((256, 256))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)[0][0]  # Model prediction
    label = "Real Image ✅" if prediction < 0.5 else "Fake Image ❌"

    return label, prediction

# Streamlit UI
st.title("🔍 Digital Image Forgery Detection")
st.write("Upload an image to check if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_uploaded = Image.open(uploaded_file)

    # Display uploaded image
    st.image(image_uploaded, caption="Uploaded Image", use_column_width=True)

    # Predict forgery
    label, confidence = predict_forgery(image_uploaded)

    # Display result
    st.write("### 🔎 Prediction Result:")
    st.write(f"**{label}** (Confidence: {confidence:.2f})")
