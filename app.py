from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

#Charger le modèle depuis MLflow
model_uri = "runs:/183919622f5942c8b98625b1d39dcb8a/model"  # Remplacez <RUN_ID> par l'ID du run MLflow.
#model = mlflow.pyfunc.load_model(model_uri)

# Initialiser l'application FastAPI
app = FastAPI(
    title="Californie Housing price prediction"
)

# Schéma pour les données d'entrée
class Input(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float

@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

# Point de terminaison pour les prédictions
@app.post("/predict")
def predict(input_data: Input):
    # Transformer les données en tableau numpy
    features = np.array([[input_data.longitude, input_data.latitude, input_data.housing_median_age,
                           input_data.total_rooms, input_data.total_bedrooms, input_data.population,
                           input_data.households, input_data.median_income]])
    # Faire une prédiction
    prediction = np.sum(features) #model.predict(features)
    return {"prediction": float(prediction)} #{"prediction": float(prediction[0])}