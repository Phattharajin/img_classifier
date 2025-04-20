
import streamlit as st

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle

with open ('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
    
st.title("Image Classification with MObileNetV2 by Phattharajin") 

upload_file = st.file_uploader("Upload image:", type=["jpg", "jpeg"])


if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption="uplaod image")
    
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]

    # Display predictions

for i, pred in enumerate(top_preds):
  st.write(f"{i+1}. **{pred[1]}** — {round(pred[2]*100, 2)}%") 
