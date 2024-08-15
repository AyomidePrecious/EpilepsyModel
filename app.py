# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:01:52 2024

@author: turningpointKS
"""

import numpy as np
import pickle
import streamlit as st

def epilepsy_prediction(input_data, loaded_model):
    # Convert input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array for a single instance prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    try:
        # Make prediction
        prediction = loaded_model.predict(input_data_reshaped)
        # Return prediction result
        if prediction[0] == 1:
            return 'The records indicate a likelihood of epilepsy'
        else:
            return 'The records do not indicate a likelihood of epilepsy'
    except Exception as e:
        return f"Error during prediction: {e}"

def main():
    # Title of the web app
    st.title('Epilepsy Prediction Web App')


    if uploaded_file:
        try:
            loaded_model = pickle.load(uploaded_file)
            st.write("Model loaded successfully")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            loaded_model = None
    else:
        st.warning("Please upload a model file.")

    # Getting the input data from the user
    minimum = st.text_input('Minimum value')
    maximum = st.text_input('Maximum value')
    mean = st.text_input('Mean value')
    standard_dev = st.text_input('Standard Deviation value')
    rms = st.text_input('RMS value')
    zcf = st.text_input('ZCF value')
    variance = st.text_input('Variance value')
    median = st.text_input('Median value')
    kurtosis = st.text_input('Kurtosis value')
    skewness = st.text_input('Skewness value')
    shannon_ent = st.text_input('Shannon Entropy value')

    # Convert inputs to floats and handle empty inputs
    input_data = []
    for value in [minimum, maximum, mean, standard_dev, rms, zcf, variance, median, kurtosis, skewness, shannon_ent]:
        try:
            input_data.append(float(value) if value else None)
        except ValueError:
            st.error("Please enter valid numeric values.")
            return

    # Creating a button for Prediction
    if st.button('Epilepsy Test Result'):
        # Check if all fields have valid inputs
        if None in input_data:
            st.error("Please enter all values.")
        elif 'loaded_model' not in locals():
            st.error("Model not loaded. Cannot make predictions.")
        else:
            # Predict and display result
            diagnosis = epilepsy_prediction(input_data, loaded_model)
            st.success(diagnosis)

if __name__ == '__main__':
    main()
