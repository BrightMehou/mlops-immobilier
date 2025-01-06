import streamlit as st
import requests
from app import Input


st.markdown("# California Housing Price Prediction")
st.write("This is a simple web app that predicts the price of a house in California based on some features. If you want to know the price of a house, please provide the following information and click on the 'Predict' button.")

def model_prediction(Input):
    url = "http://localhost:8000/predict"
    response = requests.post(url, json=Input.model_dump())
    return response.json()["prediction"]

longitude = st.text_input("Longitude", value=0.0)
latitude = st.text_input("Latitude", value=0.0)
housing_median_age = st.text_input("Housing median age", value=0.0)
total_rooms = st.text_input("Total rooms", value=0.0)
total_bedrooms = st.text_input("Total bedrooms", value=0.0)
population = st.text_input("Population", value=0.0)
households = st.text_input("Households", value=0.0)
median_income = st.text_input("Median income", value=0.0)

if st.button("Predict"):
    Input = Input(
        longitude=float(longitude), latitude=float(latitude), housing_median_age=float(housing_median_age),
        total_rooms=float(total_rooms), total_bedrooms=float(total_bedrooms), population=float(population),
        households=float(households), median_income=float(median_income)) 
    prediction = model_prediction(Input=Input)
    st.write(f"The predicted housing price is: {prediction} dollars.")