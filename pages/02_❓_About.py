# ANPR APP
# Francesco Esposito - July 2022

# Import libraries
import streamlit as st
from utility import licensePlatesMetrics, ocrMetrics

# Set page option
st.set_page_config(
    page_title='Automatic Number Plate Recognition APP',
    page_icon='🚘')

# Body
st.header('About')
st.markdown('Made with ❤️ by [Francesco Esposito](https://www.linkedin.com/in/francesco-esposito-7b275772/) - July 2022')
st.markdown('[GitHub project page](https://github.com/lupon1/automatic-number-plate-recognition-app)')
st.markdown('[MIT License](https://raw.githubusercontent.com/lupon1/automatic-number-plate-recognition-app/main/LICENSE)')
st.markdown('#')  # Space

# Algorithms metrics
col1, col2 = st.columns(2)
with col1:
    st.subheader('License Plate Model')
    vers = licensePlatesMetrics.metrics['version']
    step = licensePlatesMetrics.metrics['step'].split('/')[0]
    prec = int(round(licensePlatesMetrics.metrics['precision']*100, 0))
    rec = int(round(licensePlatesMetrics.metrics['recall']*100, 0))
    map5 = int(round(licensePlatesMetrics.metrics['mAP@.5']*100, 0))
    map95 = int(round(licensePlatesMetrics.metrics['mAP@.5:.95']*100, 0))
    st.text(f'Algorithm version: {vers}')
    st.text(f'Training epochs: {step}')
    st.text(f'Precision: {prec}%')
    st.text(f'Recall: {rec}%')
    st.text(f'mAP@.5: {map5}%')
    st.text(f'mAP@.5:.95: {map95}%')

with col2:
    st.subheader('OCR Model')
    vers = ocrMetrics.metrics['version']
    step = ocrMetrics.metrics['step'].split('/')[0]
    prec = int(round(ocrMetrics.metrics['precision']*100, 0))
    rec = int(round(ocrMetrics.metrics['recall']*100, 0))
    map5 = int(round(ocrMetrics.metrics['mAP@.5']*100, 0))
    map95 = int(round(ocrMetrics.metrics['mAP@.5:.95']*100, 0))
    st.text(f'Algorithm version: {vers}')
    st.text(f'Training epochs: {step}')
    st.text(f'Precision: {prec}%')
    st.text(f'Recall: {rec}%')
    st.text(f'mAP@.5: {map5}%')
    st.text(f'mAP@.5:.95: {map95}%')
