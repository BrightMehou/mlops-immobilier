from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
from mlflow import MlflowClient
import numpy as np
from pprint import pprint
#Charger le modèle depuis MLflow
#model_uri = "runs:/636da9c2964f4fe9a577a19c48e25c7d/model"  # Remplacez <RUN_ID> par l'ID du run MLflow.

# client = MlflowClient()
# experiment = client.get_experiment_by_name("imo")
# run = client.search_runs(experiment.experiment_id)[0]
# model_uri = f"runs:/{run.info.run_id}/model"  # Remplacez <RUN_ID> par l'ID du run MLflow.
# model = mlflow.pyfunc.load_model(model_uri)

model_name = "sk-learn-random-forest-reg-model"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
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
    prediction = model.predict(features) #np.sum(features) 
    return {"prediction": float(prediction[0])} #{"prediction": float(prediction)} #