import streamlit as st
import requests
import numpy as np
import os
from PIL import Image
from roboflow import Roboflow
from tempfile import NamedTemporaryFile

banner_path = "https://upload.wikimedia.org/wikipedia/commons/6/69/Hereditary_Spherocytosis_smear_2010-03-17.JPG"
lw_logo_path = "https://img.evbuc.com/https%3A%2F%2Fcdn.evbuc.com%2Fimages%2F350364489%2F178548290987%2F1%2Foriginal.20220908-095656?w=512&auto=format%2Ccompress&q=75&sharp=10&rect=0%2C0%2C512%2C512&s=1fc76695be9e39dd903a1a2fa8197d2a"
custom_height = 200

ricardo_path = "https://res.cloudinary.com/wagon/image/upload/c_fill,g_face,h_200,w_200/v1682162684/g9y5h3tijuolveoqi9cs.jpg"
claudia_path = "https://avatars.githubusercontent.com/u/19194859?v=4"
afonso_path = "https://media.licdn.com/dms/image/C4D03AQHFoTNrs6zvDQ/profile-displayphoto-shrink_800_800/0/1516951814538?e=1701907200&v=beta&t=HZFhy2Wm8NsjGUBs5l5lRTZzbtroNzDPyJm5uNPyrQQ"
mourad_path = "https://res.cloudinary.com/wagon/image/upload/c_fill,g_face,h_200,w_200/v1683049623/v6ip6iigzl2mntfeykms.jpg"
alvaro_path = "https://yt3.googleusercontent.com/paZVe7N1ObfG9XyrwKV4TyROhvHM0whWNU8xiVheQEspQR8mMoJ6rnzR3QL3xEgsRBoJ6hQ29w=s176-c-k-c0x00ffffff-no-rj"

st.markdown(
    f"""
    <style>
    .image-container {{
        height: {custom_height}px;
        overflow: hidden;
    }}
    .image-container img {{
        width: 100%;
        height: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(f'<div class="image-container"><img src="{banner_path}"></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["HS Detector", "About"])

with tab1:

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

with tab2:
    st.title("About HS Detector")
    st.markdown("""
    <div style="text-align: justify;">
    <b>HS Detector Web App is a tool developed for early detection of Hereditary Spherocytosis (HS) online</b> : you only need to upload a blood smear image of a patient on our web app and, through the use of deep learning (DL) techniques and convolutional neural networks (CNN), you will receive information regarding the chances the patient may or may not have hereditary spherocytosis by checking the presence of spherocytes.
    <br></br>
    Both patients and physicians can use HS Detector!
    <br></br>
    <b>DISCLAIMER: HS Detector Web App is not meant to replace diagnosis or confirmation tests of HS. Please consult a medical professional / doctor for further testing and treatment.</b>
    </div>
    """, unsafe_allow_html=True)

    st.header("The Team")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(claudia_path, use_column_width=True)
        st.markdown("<p style='text-align:center;'><a href='https://www.linkedin.com/in/claudia-beltran-a4343740/'>Claudia Beltrán Bocanegra</a></p>", unsafe_allow_html=True)

    with col2:
        st.image(alvaro_path, use_column_width=True)
        st.markdown("<p style='text-align:center;'><a href='https://www.linkedin.com/in/alvaro-carranza/'>Alvaro Carranza</a></p>", unsafe_allow_html=True)

    with col3:
        st.image(ricardo_path, use_column_width=True)
        st.markdown("<p style='text-align:center;'><a href='https://www.linkedin.com/in/ricardo-carrillo-cruz-6b78997/'>Ricardo Carrillo Cruz</a></p>", unsafe_allow_html=True)

    with col4:
        st.image(mourad_path, use_column_width=True)
        st.markdown("<p style='text-align:center;'><a href='https://www.linkedin.com/in/mouradelmoufid/'>Mourad El Moufid</a></p>", unsafe_allow_html=True)

    with col5:
        st.image(afonso_path, use_column_width=True)
        st.markdown("<p style='text-align:center;'><a href='https://www.linkedin.com/in/afonso-vaz-pinheiro-294799130/'>Afonso Pinheiro</a></p>", unsafe_allow_html=True)

    st.header("Acknowledgments")
    st.markdown("""
    <div style="text-align: justify;">
    We thank Ricardo’s family who inspired him to reflect on his experience with HS and how he could use his work to improve the diagnosis for others.
    We thank LeWagon’s community and instructors for giving us the tools to develop these ideas.
    We thank Ido Azaria for his invaluable mentorship.
    We thank Ben Auzanneau for his commitment to the program and this project.
    Team, special acknowledgement to all of you for all the hard work.
    <br></br>
    One of the main inspirations for the project is the work of Maria Delgado-Ortet, Angel Molina, Santiago Alférez, José Rodellar, and Anna Merino (2020), who developed a three-stage pipeline to (1) segment erythrocytes, (2) crop and mask them, and (3) classify them into malaria-infected or not.
    One of the datasets used during the training (Dataset A) was in fact also collected by this research team at the Hospital Clínic of Barcelona.
    </div>
    """, unsafe_allow_html=True)


st.divider()

lw_logo_html = f"""
<img src="{lw_logo_path}" alt="Custom Image" width="50">
"""

st.markdown(lw_logo_html, unsafe_allow_html=True)
st.write("")
st.markdown("**A project made by [Le Wagon](https://www.lewagon.com/)'s Data Science Batch #1275.**")
st.markdown("*Visit the [project's repository](https://github.com/rcarrillocruz/hereditary_spherocytosis_detection) for more information!*")
