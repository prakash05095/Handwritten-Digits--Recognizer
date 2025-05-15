import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load the trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

# Page layout
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Handwritten Digit Recognizer")
st.write("Upload a digit image (28x28 pixels, black on white background).")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Process the image
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    image = ImageOps.invert(image)                  # Invert for white digit on black
    image = image.resize((28, 28))                  # Resize to 28x28

    # Normalize and reshape
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict digit
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Display results
    st.image(image, caption="Processed Image", width=150)
    st.subheader(f"Predicted Digit: {predicted_digit}")
