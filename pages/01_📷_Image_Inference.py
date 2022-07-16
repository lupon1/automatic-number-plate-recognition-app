# ANPR APP
# Francesco Esposito - July 2022

# Import libraries
import streamlit as st
from PIL import Image
import time
from utility import detect
import numpy as np
from os import path
import base64

# Variable declaration
workingPath = path.dirname(__file__)
loadingImage = path.join(workingPath, '..', 'utility', 'graphics', 'loadingBig.gif')
selectFileImage = path.join(workingPath, '..', 'utility', 'graphics', 'selectFile.png')

# Loading images
with open(loadingImage, "rb") as gif:
    contents = gif.read()
    data_url = base64.b64encode(contents).decode("utf-8")
selectFileImage = Image.open(selectFileImage)

# Set page option
st.set_page_config(
    page_title='Automatic Number Plate Recognition APP',
    page_icon='🚘')

# Body
imageSlot = st.empty()
imageSlot.image(selectFileImage)
textSlot = st.empty()
textSlot.markdown('⬅️ **Upload an image with sidebar widget**')

# Sidebar
text_number_plate = st.sidebar.markdown('Predicted number plate: ⛔')
text_ocr_conf = st.sidebar.markdown('OCR Confidence: ⛔')
text_inf_time = st.sidebar.markdown('Inference time: ⛔')
st.sidebar.markdown('---')
uploaded_file = st.sidebar.file_uploader('')

# When a image is uploaded
if uploaded_file is not None:
    # Show loading gif
    textSlot.empty()
    imageSlot.markdown(f'<p align="center"><img src="data:image/gif;base64,{data_url}"></p>', unsafe_allow_html=True)
    time.sleep(1.75)
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    predImage, numberPlate, confMean, inferenceTime = detect.plate_from_photo(img_array)
    # Show inference results
    imageSlot.image(predImage)
    textSlot.markdown(f'Inference time: **{inferenceTime}ms** - OCR Confidence: **{confMean}%** - Predicted number plate: **{numberPlate}**')
    text_number_plate.markdown(f'Predicted number plate: **{numberPlate}**')
    text_ocr_conf.markdown(f'OCR Confidence: **{confMean}%**')
    text_inf_time.markdown(f'Inference time: **{inferenceTime}ms**')
