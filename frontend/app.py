import streamlit as st
import requests
import numpy as np
from PIL import Image

st.title('Welcome to Hereditary Spherocytosis Detector App')

url = 'https://hs-detector-pyur4ofpea-ew.a.run.app/v1/models/hs-retinanet:predict'

uploaded_file = st.file_uploader("Please upload a blood smear picture...",
                    type=['png', 'jpeg', 'jpg'],
                )

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    payload = {"instances": [image_np.tolist()]}

    result=requests.post(url, json=payload)
    result = result.json()

    st.write(result)
