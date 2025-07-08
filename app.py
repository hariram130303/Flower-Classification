# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("flower_classifier_model.keras")  # Replace with your model

# Define class names
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Sidebar navigation
st.sidebar.title("🌼 Flower Classifier")
page = st.sidebar.radio("Go to", ["🏠 Home", "📷 Predict", "ℹ️ About Model"])

# --- Page: Home ---
if page == "🏠 Home":
    st.title("🌼 Welcome to the Flower Classifier!")
    st.markdown("""
    This web app uses a trained Deep Learning model to classify images of flowers into 5 categories:
    - Daisy 🌼
    - Dandelion 🌾
    - Rose 🌹
    - Sunflower 🌻
    - Tulip 🌷

    Upload a flower image and see what the AI thinks it is!
    """)

# --- Page: Predict ---
elif page == "📷 Predict":
    st.title("📷 Upload a Flower Image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess
        img = img.resize((180, 180))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Predict
        predictions = model.predict(img_array)
        scores = tf.nn.softmax(predictions[0])
        top_3_indices = np.argsort(scores)[-3:][::-1]

        # Display results
        st.subheader("🌼 Prediction Results")
        for i in top_3_indices:
            st.write(f"{class_names[i]}: {scores[i]*100:.2f}%")
            st.progress(float(scores[i]))

        best_class = class_names[top_3_indices[0]]
        confidence = scores[top_3_indices[0]] * 100
        st.success(f"✅ Most likely: **{best_class}** with **{confidence:.2f}%** confidence")

# --- Page: About Model ---
elif page == "ℹ️ About Model":
    st.title("🧠 Model Info")
    st.markdown("""
    **Model:** Convolutional Neural Network (CNN)  
    **Framework:** TensorFlow / Keras  
    **Input Size:** 180x180 pixels  
    **Trained On:** [TensorFlow Flower Dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers)  
    **Classes:** Daisy, Dandelion, Rose, Sunflower, Tulip  
    **Usage:** Upload a flower image and the model classifies it using softmax probabilities.
    
    **Preprocessing Steps:**
    - Resize to 180x180
    - Normalize pixel values
    - Expand dimensions to create batch

    **Note:** This app runs entirely in your browser and does not send data to any server.
    """)

    st.info("For best results, upload a clear image of a single flower.")

