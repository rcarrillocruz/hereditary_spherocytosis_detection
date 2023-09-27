import streamlit as st
import requests
import numpy as np
import os
from PIL import Image
from roboflow import Roboflow
from tempfile import NamedTemporaryFile

st.title('Welcome to Hereditary Spherocytosis Detector App')

model = 'roboflow-30-instance-segmentation'
url = 'https://hs-detector-pyur4ofpea-ew.a.run.app/v1/models/hs-retinanet:predict'

uploaded_file = st.file_uploader("Please upload a blood smear picture...",
                    type=['png', 'jpeg', 'jpg'],
                )

if uploaded_file is not None:
    if model == 'hs-retinanet':
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        payload = {"instances": [image_np.tolist()]}

        result=requests.post(url, json=payload)
        result = result.json()

        st.write(result)
    elif model == 'roboflow-30-instance-segmentation':
        rf = Roboflow(api_key=os.environ['API_KEY'])
        project = rf.workspace().project('hereditary-spherocytosis')
        model = project.version(2).model

        # infer on a local image
        with NamedTemporaryFile(dir='.') as f:
            f.write(uploaded_file.getbuffer())
            prediction = model.predict(f.name)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(prediction.plot())
            #prediction.save('prediction.jpg')
            #st.image('prediction.jpg')
            with st.expander("See prediction:"):
                st.write(prediction.json())

        # visualize your prediction
       # model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")
