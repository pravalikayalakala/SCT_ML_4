import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load trained model
model = tf.keras.models.load_model("hand_gesture_cnn.h5")

# Load class labels
with open("gesture_labels.json", "r") as f:
    gesture_labels = json.load(f)
    gesture_labels = {int(v): k for k, v in gesture_labels.items()}

st.title("🤚 Hand Gesture Recognition")
st.write("Upload an image of a hand gesture, and the model will predict it.")

# File uploader
uploaded_file = st.file_uploader("Choose a hand gesture image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Gesture", use_column_width=True)

    # Preprocess image
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.success(f"Predicted Gesture: **{gesture_labels[predicted_class]}** ({confidence*100:.2f}%)")
