import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

model = load_model("model.h5")

CATEGORIES = [
    "Normal Note",
    "Torn Note",
    "Stained Note",
    "Faded Note",
    "Suspicious/Fake Note"
]

IMG_SIZE = 128

st.title("AI Based Currency Note Detection")

uploaded_file = st.file_uploader(
    "Upload Currency Note Image",
    type=["jpg", "png", "jpeg"]
)
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = np.array(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    class_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    result = CATEGORIES[class_index]
    st.success(f"Prediction: {result}")

    st.info(f"Confidence: {confidence:.2f}%")