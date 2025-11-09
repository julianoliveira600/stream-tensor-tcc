import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

# --- 1. Carregar o Modelo (Sua função, sem alterações) ---
@st.cache_resource
def carrega_modelo():
    """Baixa o modelo do GDrive e o carrega na memória."""
    
    # ID do seu arquivo no Google Drive
    url = 'https://drive.google.com/uc?id=1bfYFaJwa2CKQ9jz8mKtAQuFU3kmfKDiW'
    output_path = 'modelo_quantizado_float16.tflite'
    
    # Baixar se não existir
    try:
        with open(output_path, "rb") as f:
            pass
        st.write("Modelo já está em cache.")
    except FileNotFoundError:
        st.write("Baixando modelo do Google Drive...")
        gdown.download(url, output_path, quiet=False)
        st.write("Download concluído.")

    # Carregar o interpretador TFLite
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    return interpreter

# --- 2. Carregar e Pré-processar a Imagem (MUITAS CORREÇÕES AQUI) ---
def carrega_e_prepara_imagem(interpreter):
    """Lida com o upload e o pré-processamento completo da imagem."""
    
    uploaded_file = st.file_uploader('Arraste e solte uma imagem ou clique aqui para selecionar uma', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image_pil = Image.open(io.BytesIO(image_data))

        st.image(image_pil, caption="Imagem Enviada")
        st.success('Imagem foi carregada com sucesso')

        # --- Início das Correções Críticas ---

        # CORREÇÃO 1: Redimensionar a imagem para o tamanho da entrada do modelo
        # O modelo foi treinado em (256, 256)
        image_resized = image_pil.resize((256, 256))

        # CORREÇÃO 2: Converter para array numpy
        # (O seu código tinha um erro de digitação: 'dtype.float32')
        image_array = np.array(image_resized, dtype=np.float32)

        # CORREÇÃO 3: Adicionar a dimensão do "lote" (batch)
        image_batch = np.expand_dims(image_array, axis=0)

        # CORREÇÃO 4: Aplicar o pré-processamento InceptionV3
        # Esta é a etapa mais importante que estava faltando.
        # Ela converte os pixels de [0, 255] para [-1, 1].
        # O seu código antigo (image / 255.0) estava errado para este modelo.
        image_preprocessed = tf.keras.applications.inception_v3.preprocess_input(image_batch)

        # CORREÇÃO 5: Garantir o tipo de dado final (float16)
        # O modelo TFLite foi quantizado e espera float16, não float32.
        input_details = interpreter.get_input_details()
        input_dtype = input_details[0]['dtype']
        
        if input_dtype == np.float16:
            image_final = image_preprocessed.astype(np.float16)
        else:
            image_final = image_preprocessed
        
        # --- Fim das Correções ---

        return image_final
    
    return None

# --- 3. Fazer a Previsão (CORREÇÃO DE CLASSES AQUI) ---
def previsao(interpreter, image):
    """Executa o modelo e exibe os resultados."""
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define o tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], image)

    # Executa a inferência
    interpreter.invoke()

    # Pega os resultados
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # CORREÇÃO 6: A ordem das classes estava invertida.
    # O modelo aprendeu 'benign' como 0 e 'malignant' como 1.
    classes = ['Benigno', 'Maligno']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]
    
    # Arredondar para 2 casas decimais
    df['probabilidades (%)'] = df['probabilidades (%)'].round(2)

    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', 
                 text='probabilidades (%)', title='Probabilidade de Câncer Mamário')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)

# --- 4. Função Principal (SEM ERROS AQUI) ---
def main():
    
    st.set_page_config(
        page_title="Classificador de Câncer Mamário",
        page_icon="🔬",
    )
    
    st.title("🔬 Classificador de Câncer Mamário em Animais")
    st.write("""
    Este aplicativo utiliza um modelo de Deep Learning (InceptionV3) 
    para classificar se uma imagem histopatológica indica um tumor 
    Benigno ou Maligno.
    """)

    # Carrega modelo
    interpreter = carrega_modelo()
    
    # Carrega Imagem e a pré-processa
    image_para_modelo = carrega_e_prepara_imagem(interpreter)

    # Classifica
    if image_para_modelo is not None: 
        previsao(interpreter, image_para_modelo)

# --- 5. Ponto de Entrada (O ERRO PRINCIPAL ESTAVA AQUI) ---

# CORREÇÃO 7: O erro de digitação
# Você escreveu "__name__" (com 'n'). O correto é "__main__" (com 'm').
# Por isso o seu código "nao carrega".
if __name__ == "__main__":
    main()