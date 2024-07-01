import streamlit as st
import joblib
import numpy as np

# Load the trained model from the .pkl file
model = joblib.load('linear_regression_model.pkl')

# Streamlit app title
st.title('Concrete Compressive Strength Prediction')

# Input fields for the user to enter feature values
cement = st.number_input('Cement (kg in a m^3 mixture)', min_value=0.0)
blast_furnace_slag = st.number_input('Blast Furnace Slag (kg in a m^3 mixture)', min_value=0.0)
fly_ash = st.number_input('Fly Ash (kg in a m^3 mixture)', min_value=0.0)
water = st.number_input('Water (kg in a m^3 mixture)', min_value=0.0)
superplasticizer = st.number_input('Superplasticizer (kg in a m^3 mixture)', min_value=0.0)
coarse_aggregate = st.number_input('Coarse Aggregate (kg in a m^3 mixture)', min_value=0.0)
fine_aggregate = st.number_input('Fine Aggregate (kg in a m^3 mixture)', min_value=0.0)
age = st.number_input('Age (days)', min_value=0)

# Predict button
if st.button('Predict Concrete Compressive Strength'):
    # Collect user inputs into a feature array
    features = np.array([[cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])
    
    # Predict using the loaded model
    prediction = model.predict(features)
    
    # Display the prediction
    st.success(f'Predicted Concrete Compressive Strength: {prediction[0]:.2f} MPa')
