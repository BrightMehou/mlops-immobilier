from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np
import pandas as pd
from pydantic import BaseModel


# Schéma pour représenter les données d'entrée sous forme structurée
class Input(BaseModel):
    medinc: float  # Revenu médian des ménages
    houseage: float  # Âge moyen des maisons
    averooms: float  # Nombre moyen de pièces par logement
    avebedrms: float  # Nombre moyen de chambres par logement
    population: float  # Population de la région
    aveoccup: float  # Nombre moyen d'occupants par logement
    latitude: float  # Latitude de la région
    longitude: float  # Longitude de la région


# Nom et version du modèle à charger depuis MLflow
model_name = "Production-model"
model_version = 1

# Charger le modèle MLflow spécifié
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# Initialiser l'application FastAPI
app = FastAPI(
    title="Prédiction des prix des logements en Californie",
    description="API simple pour prédire les prix des logements en Californie",
    version="0.1.0",
)


@app.get("/")
async def read_main() -> dict:
    """
    Point de terminaison racine de l'API.

    Retourne un message de bienvenue pour indiquer que l'API fonctionne.
    """
    return {"msg": "Hello World"}


@app.post("/predict")
def predict(input_data: Input) -> dict:
    """
    Point de terminaison pour effectuer une prédiction des prix des logements.

    Args:
        input_data (Input): Données d'entrée structurées contenant les caractéristiques nécessaires pour la prédiction.

    Returns:
        dict: Prédiction du prix du logement sous la forme d'un dictionnaire.
    """
    # Convertir les données d'entrée en tableau NumPy
    features = np.array(
        [
            [
                input_data.medinc,
                input_data.houseage,
                input_data.averooms,
                input_data.avebedrms,
                input_data.population,
                input_data.aveoccup,
                input_data.latitude,
                input_data.longitude,
            ]
        ]
    )

    # Transformer les données en DataFrame pour correspondre au format attendu par le modèle
    features = pd.DataFrame(
        features,
        columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
    )

    # Effectuer une prédiction avec le modèle chargé
    prediction = model.predict(features)

    # Retourner le résultat sous forme de JSON
    return {"prediction": float(prediction[0])}
