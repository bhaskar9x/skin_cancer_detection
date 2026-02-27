import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Page configuration
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

st.title("🩺 Skin Cancer Detection System")
st.write("Upload a skin image to check prediction.")

# Load model (only once)
@st.cache_resource
def load_my_model():
    model = load_model("skin_cancer_model.h5")
    return model

model = load_my_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    # Prediction
    prediction = model.predict(img)

    confidence = float(prediction[0][0])

    st.write("### Prediction Result:")

    if confidence > 0.5:
        st.error(f"⚠ Cancer Detected")
        st.write(f"Confidence: {confidence*100:.2f}%")
    else:
        st.success("✅ No Cancer Detected")
        st.write(f"Confidence: {(1-confidence)*100:.2f}%")