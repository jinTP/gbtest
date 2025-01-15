import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib


model=joblib.load('model.pkl')
st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    car_ID = st.sidebar.slider('car_ID', 4.3, 7.9, 5.4)
    wheelbase = st.sidebar.slider('wheelbase', 2.0, 4.4, 3.4)
    carlength = st.sidebar.slider('carlength', 1.0, 6.9, 1.3)
    carwidth = st.sidebar.slider('carwidth', 0.1, 2.5, 0.2)

    data = {'car_ID': car_ID,
            'wheelbase': wheelbase,
            'carlength': carlength,
            'carwidth': carwidth}
    features = pd.DataFrame(data, index=[0])
    return features.to_numpy()
#add a to_numpy() here ^


df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


prediction = model.predict(df)

st.subheader('Prediction')
st.write(prediction[0])
#st.write(prediction)

