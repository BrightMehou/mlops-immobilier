import streamlit as st
import requests

st.markdown("# Prédiction des prix des logements en Californie")
st.write(
    "Cette application web prédit le prix d'une maison en Californie en fonction de certaines caractéristiques. "
    "Veuillez fournir les informations demandées ci-dessous, puis cliquez sur le bouton 'Prédire'."
)


def model_prediction(input: dict) -> str:
    """
    Envoie les données au modèle via une requête POST et récupère la prédiction.
    Args:
        input (dict): Les données d'entrée au format JSON.
    Returns:
        str: La prédiction ou un message d'erreur.
    """
    url = "http://localhost:8000/predict"
    try:
        response = requests.post(url, json=input)
    except requests.exceptions.RequestException:
        return "Erreur : le modèle n'a pas pu effectuer une prédiction."
    if response.status_code != 200:
        return "Erreur : le modèle n'a pas pu effectuer une prédiction."
    prediction = response.json()["prediction"]
    return f"Le prix prédit pour le logement est : {prediction*(10**5):.0f} dollars."


medinc = st.text_input("Revenu médian des ménages", value=0.0)
houseage = st.text_input("Âge moyen des maisons", value=0.0)
averooms = st.text_input("Nombre moyen de pièces par logement", value=0.0)
avebedrms = st.text_input("Nombre moyen de chambres par logement", value=0.0)
population = st.text_input("Population de la région", value=0.0)
aveoccup = st.text_input("Nombre moyen d'occupants par logement", value=0.0)
latitude = st.text_input("Latitude de la région", value=0.0)
longitude = st.text_input("Longitude de la région", value=0.0)

# Bouton pour déclencher la prédiction
if st.button("Prédire"):
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
            "longitude": longitude,
        }
        prediction = model_prediction(input=input)
        st.write(prediction)
    except ValueError:
        st.write("Veuillez entrer des nombres valides dans tous les champs.")
