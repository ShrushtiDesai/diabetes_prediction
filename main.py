import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3) 
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

    user_data = { 'Pregnancies': pregnancies,
                  'Glucose': glucose,
                  'Blood Pressure': blood_pressure,
                  'Skin Thickness': skin_thickness,
                  'Insulin': insulin,
                  'BMI': bmi,
                  'Diabetes Pedigree Function': diabetes_pedigree_function,
                  'Age': age
                }

    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)

prediction = loaded_model.predict(user_input)

st.subheader('Prediction:')
if prediction[0]==1:
    st.write('The patient is diabetic')
else:
    st.write('The patient is not diabetic')

