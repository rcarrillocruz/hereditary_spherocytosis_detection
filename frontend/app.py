import streamlit as st
import requests

st.title('Welcome to Hereditary Spherocytosis Detector App')

url = 'https://hs-detector-pyur4ofpea-ew.a.run.app/predict'

uploaded_file = st.file_uploader("Please upload a blood smear picture...",
                    type=['png', 'jpeg', 'jpg'],
                )

if uploaded_file is not None:
    st.image(uploaded_file)

    response = requests.post(url, files={"uploaded_file": uploaded_file.getvalue()})
    result = response.json()['prediction']

    result_texts = {
        0: "The blood smear picture doesn't show spherocytes, \npatient does not seem to have spherocytosis",
        1: "The blood smear picture shows spherocytes, \npatient may have spherocytosis. \nPlease consult a doctor"
    }

    st.text(result_texts[int(result)])
