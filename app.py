import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Image Classification with MobileNetV2 by Phitchayatida")

upload_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption="Upload image")

    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Prediction
    preds = model.predict(x)
    st.write(f"Predictions (raw): {preds}")  # Check the raw output

    try:
        top_preds = decode_predictions(preds, top=3)[0]
        st.write(f"Decoded predictions: {top_preds}")  # Check decoded predictions

        for i, pred in enumerate(top_preds):
            st.write(f"{i+1}. **{pred[1]}** — {round(pred[2]*100, 2)}%")
    except Exception as e:
        st.write(f"Error during decoding: {str(e)}")
