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
            st.caption("Bounding boxes: Purple=Platelet  Yellow=RBC  Red=Spherocyte  Cyan=WBC")

            spherocytes = 0
            for p in prediction.json()['predictions']:
                if p['class'] == 'Spherocyte':
                    spherocytes += 1
            if spherocytes > 0:
                st.write("The model has detected {} spherocytes in the image. It is possible the patient has Hereditary Spherocytosis, please consult a doctor".format(spherocytes))
            else:
                st.write("The model has not detected spherocytes in the image. It is unlikely the patient has Hereditary Spherocytosis")

            with st.expander("See prediction result:"):
                st.write(prediction.json())
