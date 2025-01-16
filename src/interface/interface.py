import streamlit as st
import requests

st.markdown("# California Housing Price Prediction")
st.write("This is a simple web app that predicts the price of a house in California based on some features. If you want to know the price of a house, please provide the following information and click on the 'Predict' button.")

def model_prediction(input: dict):
    url = "http://localhost:8000/predict"
    try:
        response = requests.post(url, json=input)
    except requests.exceptions.RequestException:
        return "Error: the model could not make a prediction."
    if response.status_code != 200:
        return "Error: the model could not make a prediction."
    prediction = response.json()["prediction"]
    return f"The predicted housing price is: {prediction} dollars."


medinc = st.text_input("Median income", value=0.0)
houseage = st.text_input("House age", value=0.0)
averooms = st.text_input("Average number of rooms per household", value=0.0)
avebedrms = st.text_input("Average number of bedrooms per household", value=0.0)
population = st.text_input("Population", value=0.0)
aveoccup = st.text_input("Average number of household members", value=0.0)
latitude = st.text_input("Latitude", value=0.0)
longitude = st.text_input("Longitude", value=0.0)
if st.button("Predict"):
    try:
        medinc = float(medinc)
        houseage = float(houseage)
        averooms = float(averooms)
        avebedrms = float(avebedrms)
        population = float(population)
        aveoccup = float(aveoccup)
        latitude = float(latitude)
        longitude = float(longitude)

        input = {
            "medinc": medinc,
            "houseage": houseage,
            "averooms": averooms,
            "avebedrms": avebedrms,
            "population": population,
            "aveoccup": aveoccup,
            "latitude": latitude,
            "longitude": longitude
        }
        prediction = model_prediction(input=input)
        st.write(prediction)
    except ValueError:
        st.write("Please enter valid numbers in all fields.")
   