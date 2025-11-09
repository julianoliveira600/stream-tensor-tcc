import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carrega_modelo():
    #https://drive.google.com/file/d/1bfYFaJwa2CKQ9jz8mKtAQuFU3kmfKDiW/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1bfYFaJwa2CKQ9jz8mKtAQuFU3kmfKDiW'

    gdown.download(url, 'modelo_quantizado_float16.tflite')
    interpreter = tf.lite.Interpreter(model_path = 'modelo_quantizado_float16.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem ou clique aqui para selecionar uma' , type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Imagem foi carregada com sucesso')

        image = np.array(image, dtype.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image
    
def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['Maligno', 'Benigno']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    fig = px.bar(df,y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)', 
                 title='Probabilidade de Cancer Mamário')
    st.plotly_chart(fig)

def main():
    
    st.set_page_config(
        page_title="Classifica Tipos de Cancer Mamário",
        page_icon="",
    )
    
    st.write("# Classifica Tipos de Cancer Mamário")
    #Carrega modelo
    interpreter = carrega_modelo()
    #Carrega Imagem
    image = carrega_imagem()

    #Classifica
    if image is not None: 
        previsao(interpreter, image)

if __name__=="__name__":
    main()

