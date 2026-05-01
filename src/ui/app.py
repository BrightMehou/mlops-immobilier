"""
Application Streamlit pour le déploiement d’un modèle de prédiction des prix
des logements en Californie. Elle fournit une interface utilisateur interactive
pour saisir les caractéristiques d’un logement, interroger l’API de prédiction
et visualiser les explications SHAP associées.
"""

import logging
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="California housing prices", page_icon="🏠", layout="wide"
)

st.title("🏠 Prédiction du prix des logements en Californie")
st.markdown(
    """
Cette application utilise un modèle de machine learning pour prédire le prix des logements en Californie 
en fonction de plusieurs caractéristiques socio-démographiques et géographiques.
"""
)

MODEL_URL: str = os.getenv("MODEL_URL", "http://localhost:8000/predict")


def model_prediction(input: dict):
    """
    Envoie les données au modèle via une requête POST et retourne la prédiction formatée
    ainsi que les valeurs SHAP pour l’explication.

    Args:
        input (dict): Données du logement à prédire.

    Returns:
        tuple: Message textuel avec le prix prédit, et liste des valeurs SHAP.
    """

    logger.info(f"Envoi des données au modèle : {input}")
    try:
        response = requests.post(MODEL_URL, json=input)
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de connexion au modèle : {e}")
        return "❌ Erreur : impossible de contacter le modèle.", None

    if response.status_code != 200:
        logger.error(
            f"Réponse invalide du modèle ({response.status_code}) : {response.text}"
        )
        return "⚠️ Erreur : le modèle a retourné une réponse incorrecte.", None

    result = response.json()
    prediction = result["prediction"][0]
    shap_values = result.get("shap_values", [[]])[0]

    logger.info(f"Réponse reçue du modèle : {prediction} avec SHAP {shap_values}")

    text_output = (
        f"💰 Le prix prédit pour le logement est : **{prediction * (10**5):,.0f} $**."
    )
    return text_output, shap_values


st.subheader("🧾 Entrez les caractéristiques du logement")
col1, col2 = st.columns(2)
with col1:
    medinc = st.number_input(
        "💰 Revenu médian des ménages (en dizaines de milliers de $)",
        min_value=0.0,
        value=0.0,
    )
    houseage = st.number_input(
        "📅 Âge moyen des maisons (en années)", min_value=0.0, value=0.0
    )
    averooms = st.number_input(
        "🏠 Nombre moyen de pièces par logement", min_value=0.0, value=0.0
    )
    avebedrms = st.number_input(
        "🛏️ Nombre moyen de chambres par logement", min_value=0.0, value=0.0
    )
with col2:
    population = st.number_input("👥 Population de la région", min_value=1.0, value=1.0)
    aveoccup = st.number_input(
        "👨‍👩‍👧‍👦 Nombre moyen d'occupants par logement", min_value=0.0, value=0.0
    )
    latitude = st.number_input(
        "📍 Latitude de la région", min_value=31.0, max_value=43.0, value=37.0
    )
    longitude = st.number_input(
        "🗺️ Longitude de la région", min_value=-125.0, max_value=-113.0, value=-119.0
    )

bouton = st.button("📈 Prédire")
if bouton:
    input_data: dict[str, float] = {
        "MedInc": medinc,
        "HouseAge": houseage,
        "AveRooms": averooms,
        "AveBedrms": avebedrms,
        "Population": population,
        "AveOccup": aveoccup,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    logger.info("Formulaire soumis par l'utilisateur.")
    prediction_text, shap_values = model_prediction(input_data)

    shap_values = [round(val * 10**5, 0) for val in shap_values]
    if "Erreur" in prediction_text:
        st.error(prediction_text)
    else:
        st.success(prediction_text)

    if shap_values:
        feature_names: list[str] = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
        shap_df = pd.DataFrame([shap_values], columns=feature_names)
        shap_df = shap_df.melt(var_name="Feature", value_name="SHAP value")

        fig = px.bar(
            shap_df,
            x="Feature",
            y="SHAP value",
            title=f"Importance des features {sum(shap_values):,.0f} par rapport au prix de base moyen de 200,000 $",
        )
        st.plotly_chart(fig)
