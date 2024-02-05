from fastapi import FastAPI, HTTPException
import streamlit as st
import json
import requests
import pandas as pd

import streamlit as st

# CSS
css = """
<style>
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius:20px;
        border:none;
        padding: 10px 24px;
        cursor: pointer;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
"""
app = FastAPI()
st.markdown(css, unsafe_allow_html=True)
st.title('Big Data API Deployment')

if st.button('Try Me'):
    st.write('UwOOF')

# Endpoint
API_URL = "http://localhost:8080/predict"
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    # Lire le fichier CSV
    df = pd.read_csv(uploaded_file)
    st.write(df)

    # Bouton
    if st.button('Predict'):
        json_data = df.to_dict(orient='records')
        data = {"features": json_data}
        st.write(data)
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            # Afficher la prédiction
            prediction = response.json()
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Failed to get response from API : Error code ", response.status_code)
