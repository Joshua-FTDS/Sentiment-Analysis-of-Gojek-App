import pickle
import streamlit as st
from sklearn.pipeline import Pipeline
import pandas as pd
from PIL import Image
import os

# Load model yang telah disimpan
with open('svc_pipeline.pkl', 'rb') as file:
    svc_pipeline = pickle.load(file)

# Load gambar
image_path = 'gojek.jpg'
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.image(image, caption='Photo by: poin.win')
else:
    st.warning(f"Image {image_path} not found")

def predict_sentiment(text):
    text_df = pd.DataFrame({'text_processed': [text]})
    
    try:
        # Predict
        prediction = svc_pipeline.predict(text_df)[0]
        
        sentiment_map = {0: 'Negatif', 1: 'Positif'}
        sentiment = sentiment_map[prediction]
        
        return sentiment
    except Exception as e:
        print("Error details:", str(e))
        raise

# Streamlit
st.title('Analisis Sentimen Gojek')
st.write('Masukkan ulasan atau komentar tentang Gojek')

user_input = st.text_area('Teks Ulasan')

if st.button('Prediksi Sentimen'):
    if user_input:
        sentiment = predict_sentiment(user_input)
        
        st.write(f'Sentimen: {sentiment}')
        
        if sentiment == 'Positif':
            st.success('Ulasan Positif!')
        else:
            st.error('Ulasan Negatif')
    else:
        st.warning('Silakan masukkan teks terlebih dahulu')