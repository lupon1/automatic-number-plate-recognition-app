# ANPR APP
# Francesco Esposito - July 2022

# Import libraries
import streamlit as st
from os import path
from PIL import Image

# Variables
workingPath = path.dirname(__file__)
cover = Image.open(path.join(workingPath, 'utility', 'graphics', 'cover.png'))

# Set page option
st.set_page_config(
    page_title='Automatic Number Plate Recognition APP',
    page_icon=':oncoming_automobile:')

# Body
st.header('Automatic Number Plate Recognition APP')
st.markdown('''
Automatic number plate recognition (ANPR or LPR) is a surveillance and access control
method that uses optical character recognition in images to read vehicle license plates.
It first uses a series of image manipulation techniques to detect,
normalize and enhance the image of the license plate number,
and finally optical character recognition to extract the alphanumerics from the license plate.
ANPR technology tends to be region specific, due to the variation between license plates from place to place
but in this case I have try to train a unique model for many countries.
''')
st.image(cover)

from utility import detect
